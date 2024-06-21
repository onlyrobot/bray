from typing import List, Dict, Tuple, Union
import asyncio, time, random, os
import pickle, struct
from concurrent.futures import ThreadPoolExecutor
import multiprocessing.shared_memory as shm

import ray
from bray.utils import ray_scheduling_local
import torch
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    handle_nested_array,
    flatten_nested_array,
    unflatten_nested_array,
    make_batch,
    split_batch,
    serialize_nested_array,
    deserialize_nested_array,
)
from bray.utils.worker import WorkerLoadBalance
from bray.master.master import merge, query, merge_time_ms


def get_torch_model_weights(model: torch.nn.Module) -> Dict:
    return {k: v.cpu().detach().numpy() 
        for k, v in model.state_dict().items()}


def set_torch_model_weights(model: torch.nn.Module, weights: Dict):
    model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in weights.items()})


def save_weights(weights: Dict, path: str):
    torch.save({k: torch.as_tensor(v) for k, v in weights.items()}, path)


class ModelForwardProxy:
    def __init__(self, max_batch_size, parallel=1):
        self.max_batch_size, self.parallel = max_batch_size, parallel
        self.forward_conds = [
            asyncio.Condition() for _ in range(1 + parallel)]
        self.forward_cond = None
        self.pending_forwards, self.ready_forwards = [], []
        self.pending_time, self.forward_beg = 0.0, time.time()

    async def proxy_forward(self, args, kwargs) -> NestedArray:
        raise NotImplementedError

    async def batch_forward(self, pending_forwards: List):
        self.forward_beg, parts = time.time(), []
        forward_beg = self.forward_beg
        args, kwargs = pending_forwards[0]
        if len(pending_forwards) > 1:
            args, kwargs = make_batch(pending_forwards, 
                concate=True, parts=parts)
        try: outputs = await self.proxy_forward(args, kwargs)
        except Exception as e:
            print(f"Fail to forward: {e}")
            return [None] * len(pending_forwards)
        ready_forwards = [outputs]
        if len(pending_forwards) > 1:
            ready_forwards = split_batch(outputs, parts)
        # merge_time_ms(
        #     f"forward/{self.name}", forward_beg, mode="proxy")
        self.pending_time = self.pending_time * 7 / 8 + (
            time.time() - forward_beg) / 8 / self.parallel
        # merge(f"wait/parallel{self.parallel}", self.pending_time)
        return ready_forwards

    async def _forward_coro(self, forward_cond):
        async with forward_cond:
            await forward_cond.wait_for(lambda: self.pending_forwards)
        pending_size = len(self.pending_forwards)
        last_pending_size = 0
        while (pending_size < self.max_batch_size 
        and pending_size != last_pending_size):
            last_pending_size = pending_size
            await asyncio.sleep(0)
            pending_size = len(self.pending_forwards)
        pending_forwards = self.pending_forwards
        ready_forwards = self.ready_forwards
        self.pending_forwards, self.ready_forwards = [], []
        self.forward_cond = None
        # set the ready_forwards to the previous pending_forwards
        ready_forwards[:] = await self.batch_forward(pending_forwards)
        ready_forwards.append(len(pending_forwards)) # forward done cnt
        async with forward_cond:
            forward_cond.notify_all()
        while ready_forwards[-1] != 0: await asyncio.sleep(0)
        self.forward_conds.append(forward_cond)

    async def _initialize_forward_coro(self):
        while not self.forward_conds: await asyncio.sleep(0.001)
        if self.forward_cond: return
        self.forward_cond = self.forward_conds.pop()
        pending_time = self.forward_beg + self.pending_time - time.time()
        if pending_time > 0.001:
            await asyncio.sleep(pending_time)
        asyncio.create_task(self._forward_coro(self.forward_cond))

    async def forward(self, args, kwargs, pending=True) -> NestedArray:
        if self.max_batch_size < 2:
            return (await self.batch_forward([(args, kwargs)]))[0]
        while len(self.pending_forwards) >= self.max_batch_size:
            if not pending:
                raise RuntimeError("Too many requests.")
            await asyncio.sleep(0.001)
        while not self.forward_cond:
            await self._initialize_forward_coro()
        forward_index = len(self.pending_forwards)
        self.pending_forwards.append((args, kwargs))
        last_forward_cond = self.forward_cond
        last_ready_forwards = self.ready_forwards
        async with last_forward_cond:
            if forward_index == 0: last_forward_cond.notify_all()
            await last_forward_cond.wait_for(lambda: last_ready_forwards)
        last_ready_forwards[-1] -= 1
        return last_ready_forwards[forward_index]


class TorchModelWorker:
    def __init__(self, name, model: "Model", num_cpus, num_gpus, 
        raw_model, **kwargs):
        self.name, self.model = name, model
        self.num_cpus, self.num_gpus = num_cpus, num_gpus
        self.torch_model = raw_model

    def initialize_runtime(self):
        weights = ray.get(self.model.get_weights.remote(self.name))
        num_cpus = max(1, int(self.num_cpus))
        if torch.get_num_interop_threads() != num_cpus:
            torch.set_num_interop_threads(num_cpus)
        if torch.get_num_threads() != num_cpus: 
            torch.set_num_threads(num_cpus)
        model = self.torch_model.requires_grad_(False).eval()
        set_torch_model_weights(model, weights)
        self.torch_model = model.cuda() if self.num_gpus else model

    def handler(self, array):
        if not isinstance(array, (np.ndarray, torch.Tensor)):
            return array
        tensor = torch.as_tensor(array)
        return tensor if not self.num_gpus else tensor.cuda()

    def forward(self, batch_args, batch_kwargs):
        batch_args, batch_kwargs = handle_nested_array(
            (batch_args, batch_kwargs), self.handler)
        outputs = self.torch_model(*batch_args, **batch_kwargs)
        return handle_nested_array(outputs, 
        lambda x: x.cpu().detach().numpy(), type_check=False)

    def set_weights(self, weights: NestedArray):
        set_torch_model_weights(self.torch_model, weights)


class OnnxModelWorker:
    cached_onnx_session: Dict[str, tuple] = {}

    def __init__(self, name, model: "Model", num_cpus, num_gpus, 
        raw_model, use_onnx, **kwargs):
        self.name, self.model = name, model
        self.num_cpus, self.num_gpus = num_cpus, num_gpus
        self.onnx_model, self.use_onnx = raw_model, use_onnx

    def build_ort_session(self):
        provider = ["CUDAExecutionProvider"] if self.num_gpus else []
        num_cpus = max(1, int(self.num_cpus))
        import onnxruntime as ort

        self.onnx_model, onnx_model = None, self.onnx_model
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_cpus
        options.inter_op_num_threads = num_cpus
        outputs = ray.get(self.model.get_forward_outputs.remote())
        return ort.InferenceSession(
        onnx_model, options, provider), outputs

    def get_cached_onnx_session(self):
        if self.use_onnx != "train": return None, None
        base_name = self.name.split("/clone")[0]
        if base_name not in OnnxModelWorker.cached_onnx_session:
            return None, None
        return OnnxModelWorker.cached_onnx_session[base_name]

    def set_cached_onnx_session(self, sess, outputs):
        if self.use_onnx != "train": return
        base_name = self.name.split("/clone")[0]
        OnnxModelWorker.cached_onnx_session[base_name] = (sess, outputs)

    def initialize_runtime(self):
        sess, outputs = self.get_cached_onnx_session()
        if not sess: sess, outputs = self.build_ort_session()
        self.set_cached_onnx_session(sess, outputs)
        self.input_names = [i.name for i in sess.get_inputs()]
        self.sess, self.outputs = sess, outputs
        if self.use_onnx != "train": return
        weights = ray.get(self.model.get_weights.remote(self.name))
        weights = handle_nested_array(weights, lambda x: x.copy())
        self.weights = [weights[n] 
        for n in self.input_names if n in weights]

    def forward(self, batch_args, batch_kwargs):
        flatten_input = flatten_nested_array(
            batch_args + tuple(batch_kwargs.values()), sort_keys=True
        )
        if self.use_onnx == "train":
            flatten_input.extend(self.weights)
        inputs = dict(zip(self.input_names, flatten_input))
        # output_names = [
        #     output.name for output in self.sess.get_outputs()]
        # print(handle_nested_array(
        #     inputs, lambda x: (x.shape, x.dtype)))
        outputs = self.sess.run(None, inputs)
        return unflatten_nested_array(self.outputs, outputs)

    def set_weights(self, weights: NestedArray):
        if self.use_onnx != "train":
            return print("Set onnx weights only in train mode.")
        self.weights = [
        v.copy() for k, v in weights if k in self.input_names]


class ModelWeightsPublisherMeta:
    def __init__(self, name):
        self.shm_name = f"{name}-weights".replace("/", "-")
        self.cond = asyncio.Condition()
        self.weights, self.step, self.shm = None, 0, None

    def __del__(self): 
        if self.shm: self.shm.close(), self.shm.unlink()


class ModelWeightsPublisher:
    def __init__(self, name): self.name, self.pubs = name, {}

    def get_weights_publisher_meta(self, name):
        if pub := self.pubs.get(name, None): return pub
        self.pubs[name] = ModelWeightsPublisherMeta(name)
        return self.pubs[name]

    def create_shm(self, name, size):
        try: return shm.SharedMemory(name, create=True, size=size)
        except FileExistsError: pass
        existing_shm = shm.SharedMemory(name=name)
        existing_shm.close(), existing_shm.unlink()
        return shm.SharedMemory(name, create=True, size=size)

    async def set_weights(self, name, weights, step):
        signature, data = serialize_nested_array(weights)
        pub = self.get_weights_publisher_meta(name)
        if not (shm := pub.shm):
            pub.signature = pickle.dumps(signature)
            shm = self.create_shm(pub.shm_name, 8 + len(data))
        if 8 + len(data) != shm.size:
            return print("Invalid weights, signature not match")
        shm.buf[:8] = struct.pack('q', step)
        shm.buf[8: 8 + len(data)] = data
        pub.step, pub.shm = step, shm
        async with pub.cond: pub.cond.notify_all()

    async def subscribe_weights(self, name, current_step):
        pub = self.get_weights_publisher_meta(name)
        pred = lambda: pub.shm and pub.step != current_step
        coro = pub.cond.wait_for(pred)
        async with pub.cond:
            try: await asyncio.wait_for(coro, 10 * 60)
            except: pass
        return pub.shm_name, pub.step, pub.signature

    def get_step(self, name) -> int:
        ckpt_step = self.get_ckpt_steps(name)[-1]
        return self.get_weights_publisher_meta(name).step or ckpt_step


class ModelWeightsSubscriber:
    async def initialize_subscriber(self, name, model: "Model"):
        self.name, self.shm = name, None
        self.current_step = -1
        self.publisher = await model.get_weights_publisher.remote(
            self.name, ray.get_runtime_context().get_node_id())
        self.is_initialized = True
        asyncio.create_task(self.subscribe_weights())

    async def _subscribe_weights(self):
        weights_info = self.publisher.subscribe_weights.remote(
            self.name, self.current_step)
        subscribe_beg = time.time()
        shm_name, step, signature = await weights_info
        if not shm_name or step == self.current_step: return
        if not self.shm:
            self.signature = pickle.loads(signature)
            self.shm = shm.SharedMemory(shm_name)
        weights = deserialize_nested_array(
            self.signature, self.shm.buf[8:])
        await asyncio.get_running_loop().run_in_executor(
            ModelWorker.executor, 
            self.set_weights, weights)
        if step != struct.unpack('q', self.shm.buf[:8])[0]:
            print("Error: weights changed when subscribe")
        self.current_step = step
        merge_time_ms(f"subscribe/{self.name}", subscribe_beg)

    async def subscribe_weights(self):
        try: await asyncio.wait_for(self._subscribe_weights(), 
            timeout=random.randint(0, 10 * 60))
        except asyncio.TimeoutError: pass
        except Exception as e:
            print(f"Fail to subscribe weights for {self.name}: {repr(e)}")
            await asyncio.sleep(0.1)
        if not self.is_initialized: return
        asyncio.create_task(self.subscribe_weights())

    def set_weights(self, weights): raise NotImplementedError
    

class ModelWorker(ModelForwardProxy, ModelWeightsSubscriber):
    executor: ThreadPoolExecutor = None

    def __init__(self, name, max_batch_size, cpus_per_worker, 
        gpus_per_worker, port=None, use_onnx=None, model: "Model" = None
    ):
        super().__init__(max_batch_size, parallel=1)

        self.name, self.model = name, model or ray.get_actor(
            name.split("/clone")[0])
        self.is_initialized = False
        use_onnx, raw_model = ray.get(self.model.get_model.remote(
            self.name, use_onnx))
        if not use_onnx: raw_model = ray.get(raw_model)

        if use_onnx: RuntimeWorker = OnnxModelWorker
        else: RuntimeWorker = TorchModelWorker
        if not ModelWorker.executor:
            ModelWorker.executor = ThreadPoolExecutor(max_workers=1)
            
        self.runtime_worker = RuntimeWorker(
            self.name, self.model, 
            cpus_per_worker, gpus_per_worker, raw_model, 
            use_onnx=use_onnx)
        self.runtime_worker.initialize_runtime()

        self.set_weights = self.runtime_worker.set_weights
        if not port: return
        asyncio.create_task(self.initialize_http_server(port))

    async def proxy_forward(self, args, kwargs):
        forward_beg = time.time()
        outputs = await asyncio.get_running_loop().run_in_executor(
            ModelWorker.executor, 
            self.runtime_worker.forward, args, kwargs)
        merge_time_ms(f"forward/{self.name}", forward_beg)
        return outputs

    async def forward(self, args, kwargs, pending=True):
        is_initialized, self.is_initialized = self.is_initialized, True
        if not is_initialized:
            await self.initialize_subscriber(self.name, self.model)
        return await super().forward(args, kwargs, pending)

    def get_host_and_node_id(self):
        node_id = ray.get_runtime_context().get_node_id()
        return ray.util.get_node_ip_address(), node_id

    async def initialize_http_server(self, port):
        from bray.utils.http_server import launch_http_server
        await launch_http_server(port, self._http_forward)

    def initialize_http_session(self):
        node_id = ray.get_runtime_context().get_node_id()
        # if node_id == self.node_id: self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}/step"
        import requests
        self.sess = requests.Session()

    async def http_forward(self, args, kwargs, pending=True):
        if not hasattr(self, "sess"): self.initialize_http_session()
        data = pickle.dumps((args, kwargs, pending))
        res = await asyncio.get_running_loop().run_in_executor(
            None, self.sess.post, self.url, data)
        if res.status_code != 200: raise Exception(res.text)
        return pickle.loads(res.content)

    async def _http_forward(self, data: bytes) -> bytes:
        args, kwargs, pending = pickle.loads(data)
        return pickle.dumps(await self.forward(args, kwargs, True))


class ModelWorkerManagerMeta:
    def __init__(self, num_workers, max_batch_size, local_mode):
        self.num_workers, self.workers = num_workers, []
        self.pending_create_workers = 0
        self.max_batch_size, self.local_mode = max_batch_size, local_mode
        self.cond = asyncio.Condition()


class ModelWorkerManager:
    def __init__(
        self, cpus_per_worker, gpus_per_worker, memory_per_worker, port
    ):
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.port, self.w_metas = port, {}
        self.num_workers = len([n for n in ray.nodes() if n["Alive"]])
        asyncio.create_task(self.health_check())

    async def initialize_workers(
        self, name, num_workers, max_batch_size, local_mode):
        if name in self.w_metas: return
        self.w_metas[name] = ModelWorkerManagerMeta(
            num_workers, max_batch_size, local_mode)
        if num_workers is None:
            num_workers = self.num_workers
        create_coros = [
            self.create_worker(name) for _ in range(num_workers)]
        await asyncio.gather(*create_coros)

    async def create_worker(self, name):
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        num_workers = len(meta.workers) + meta.pending_create_workers
        max_workers = meta.num_workers or self.num_workers
        if num_workers >= max_workers: return
        if self.port: port = self.port = self.port + 1
        else: port = None
        RemoteModelWorker = ray.remote(ModelWorker).options(
            num_cpus=self.cpus_per_worker,
            num_gpus=self.gpus_per_worker,
            memory=self.memory_per_worker * 1024 * 1024,
            scheduling_strategy="SPREAD")
        worker = RemoteModelWorker.remote(
            name, meta.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, port)
        meta.pending_create_workers += 1
        is_health = await self._is_health(worker)
        meta.pending_create_workers -= 1
        if not is_health: return
        host, node_id = await worker.get_host_and_node_id.remote()
        if self.port:
            w = ModelWorker.__new__(ModelWorker)
            w.forward, w.remote = w, w.http_forward
            worker, w.worker = w, worker
        worker.host, worker.node_id, worker.port = host, node_id, port
        if (meta.num_workers is not None 
            and len(meta.workers) >= meta.num_workers): return
        meta.workers.append(worker)
        async with meta.cond: meta.cond.notify_all()

    async def _is_health(self, worker): raise NotImplementedError

    async def _health_check(self, name):
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        origin_workers_num = len(meta.workers)
        active_workers = [
            worker for worker in meta.workers[:origin_workers_num]
            if await self._is_health(worker)
        ]
        old_workers, meta.workers = meta.workers, active_workers
        meta.workers.extend(old_workers[origin_workers_num:])
        merge(f"worker/{name}", len(meta.workers), desc=
            {"time_window_avg": "smoothed model worker num"})
        if meta.num_workers is None: return
        num_workers = len(meta.workers) + meta.pending_create_workers
        if num_workers >= meta.num_workers: return
        create_coros = [self._create_worker(name) 
            for _ in range(meta.num_workers - num_workers)]
        await asyncio.create_task(*create_coros)

    async def health_check(self):
        check_coros = [self._health_check(n) for n in self.w_metas]
        await asyncio.gather(*check_coros)
        await asyncio.sleep(60)
        asyncio.create_task(self.health_check())

    async def subscribe_workers(self, name, node_id=None, cur_num=0):
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        pred = lambda: cur_num != len(self.get_workers(name, node_id))
        coro = meta.cond.wait_for(pred)
        async with meta.cond: 
            try: await asyncio.wait_for(coro, 10 * 60)
            except: pass
        return self.get_workers(name, node_id)

    def get_workers(self, name, node_id=None) -> List[ModelWorker]:
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        if not node_id: return meta.workers
        workers = [w for w in meta.workers if w.node_id == node_id]
        return workers if workers else meta.workers


class ModelCheckpointManager(ModelWeightsPublisher):
    def __init__(
        self, name, ckpt_name, trial_path, checkpoint, checkpoint_interval
    ):
        ModelWeightsPublisher.__init__(self, name)
        self.name, self.trial_path, self.ckpt_steps = name, trial_path, {}
        self.ckpt_name = ckpt_name
        self.checkpoint = checkpoint
        if not checkpoint_interval:
            asyncio.create_task(self._save_checkpoint())
        self.checkpoint_interval = checkpoint_interval

    async def _save_checkpoint(self, interval=10 * 60):
        await asyncio.sleep(interval)
        for name, meta in list(self.pubs.items()):
            ckpt_steps = self.get_ckpt_steps(name)
            if not meta.step or meta.step <= ckpt_steps[-1]: 
                continue
            ckpt_steps.append(meta.step)
            await self.save_checkpoint(
            name, meta.weights, meta.step)
        asyncio.create_task(self._save_checkpoint(interval))

    async def on_set_weights(self, name, weights, step):
        ckpt_step = self.get_ckpt_steps(name)[-1]
        if isinstance(interval:=self.checkpoint_interval, float):
            base = int(interval)
            s, interval = step // 10, interval - base
            while s: interval, s = interval * 10, s // 10
            interval = max(base, int(interval))
        if not interval or (step - ckpt_step) // interval < 1:
            return
        self.get_ckpt_steps(name).append(step)
        await self.save_checkpoint(name, weights, step)

    def build_ckpt_steps(self, name) -> List[int]:
        checkpoint = None if name != self.ckpt_name else self.checkpoint
        if isinstance(checkpoint, str): return [0]
        root_path = os.path.join(self.trial_path, name)
        ckpt_dir = os.path.join(root_path, "checkpoint")
        if not os.path.exists(ckpt_dir): ckpt_steps = []
        else: ckpt_steps = [int(ckpt.split(".")[0].split("-")[1]) 
            for ckpt in os.listdir(ckpt_dir) ]
        ckpt_steps.append(0)
        clone_steps = [int(ckpt.split(".")[0].split("-")[2]) 
            for ckpt in os.listdir(root_path) 
            if ckpt.startswith("clone-step")]
        ckpt_steps = sorted(set(ckpt_steps).union(clone_steps))
        is_valid = lambda c: checkpoint is None or c <= checkpoint
        return [c for c in ckpt_steps if is_valid(c)]

    def get_ckpt_steps(self, name=None) -> List[int]:
        if not name: return self.ckpt_steps
        if c := self.ckpt_steps.get(name, None): return c
        self.ckpt_steps[name] = self.build_ckpt_steps(name)
        return self.ckpt_steps[name]

    def load_checkpoint(self, name, step) -> NestedArray:
        root_path = os.path.join(self.trial_path, name)
        weights_path = os.path.join(
            root_path, "checkpoint", f"step-{step}.pt")
        if step == 0 and isinstance(self.checkpoint, str):
            weights_path = self.checkpoint
        if not os.path.exists(weights_path) and step == 0:
            weights_path = os.path.join(root_path, "weights.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(
            root_path, f"clone-step-{step}", "weights.pt")
        weights = torch.load(weights_path)
        return handle_nested_array(weights, lambda x: x.numpy())

    async def save_checkpoint(self, name, weights, step):
        ckpt_dir = os.path.join(self.trial_path, name, "checkpoint")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"step-{step}.pt")
        await asyncio.get_running_loop().run_in_executor(None, torch.save, 
        handle_nested_array(weights, torch.as_tensor), ckpt_path)
        

class ModelOnnxManager:
    def __init__(self, name, torch_path):
        self.name, self.torch_path = name, torch_path
        self.torch_model, self.use_onnxs = None, {}
        self.root_path = os.path.dirname(self.torch_path)
        self.outputs_path = os.path.join(
            self.root_path, "forward_outputs.pt")
        self.onnx_train_model = self.forward_outputs = None

    def _get_onnx_train_model(self):
        if self.onnx_train_model: return self.onnx_train_model
        from bray.model.onnx import export_onnx
        onnx_train_path = os.path.join(self.root_path, "train.onnx")
        print("Exporting onnx model to", onnx_train_path)
        if not self.torch_model:
            self.torch_model = torch.load(self.torch_path)
        self.forward_outputs = export_onnx(
            self.torch_model,
            onnx_train_path,
            self.forward_args,
            self.forward_kwargs,
            export_params=False)
        torch.save(self.forward_outputs, self.outputs_path)
        with open(onnx_train_path, "rb") as f:
            self.onnx_train_model = f.read()
        return self.onnx_train_model

    def _get_onnx_model(self, name, use_onnx, step, weights):
        if use_onnx == "train": return self._get_onnx_train_model()
        onnx_path = os.path.join(self.root_path, f"{name}.onnx")
        if os.path.exists(onnx_path):
            with open(onnx_path, "rb") as f: return f.read()
        from bray.model.onnx import export_onnx

        weights = weights or self.load_checkpoint(name, step)
        if not self.torch_model:
            self.torch_model = torch.load(self.torch_path)
        set_torch_model_weights(self.torch_model, weights)
        print(f"Exporting onnx model at step {step} to", onnx_path)
        self.forward_outputs = export_onnx(
            self.torch_model,
            onnx_path,
            self.forward_args,
            self.forward_kwargs,
            export_params=True,
            quantize=use_onnx == "quantize")
        if not os.path.exists(self.outputs_path):
            torch.save(self.forward_outputs, self.outputs_path)
        with open(onnx_path, "rb") as f: return f.read()

    def get_forward_outputs(self) -> NestedArray:
        if not self.forward_outputs:
            self.forward_outputs = torch.load(self.outputs_path)
        return self.forward_outputs

    async def get_onnx_model(self, name, use_onnx) -> bytes:
        step, weights = await self.get_target_step(name, -1)
        return await asyncio.get_running_loop().run_in_executor(None, 
        self._get_onnx_model, name, use_onnx, step, weights)


@ray.remote(num_cpus=0)
class Model(
    ModelWorkerManager, ModelCheckpointManager, ModelOnnxManager):
    def __init__(self, name: str,
        raw_model: torch.nn.Module = None,
        forward_args: Tuple[np.ndarray] = None,
        forward_kwargs: Dict[str, np.ndarray] = None,
        checkpoint_interval: int = None,
        checkpoint: Union[str, int] = None,
        max_batch_size: int = 1,
        num_workers: int = None,
        cpus_per_worker: float = 1.0,
        gpus_per_worker: float = 0.0,
        memory_per_worker: int = 1024,
        use_onnx: str = "",
        local_mode: bool = False,
        port: int = None,
        override_model: bool = True,
    ):
        self.trial_path = ray.get_runtime_context().namespace
        root_path = os.path.join(self.trial_path, f"{name}")
        if name[0] == "." or name[0] == "/":
            self.trial_path, root_path = "./", name.split("/clone")[0]
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1))

        ModelWorkerManager.__init__(self, cpus_per_worker, 
            gpus_per_worker, memory_per_worker, port)
        self.name = name.split("/clone")[0]
        ModelCheckpointManager.__init__(self, self.name, name,
            self.trial_path, checkpoint, checkpoint_interval)

        self.model, self.weights_publishers = None, {}
        self.torch_path = os.path.join(root_path, f"model.pt")

        if override_model := raw_model and (override_model or 
            not os.path.exists(self.torch_path)):
            torch.save(raw_model, self.torch_path)
        else:
            assert os.path.exists(self.torch_path), "Missing model"
        args_path = os.path.join(root_path, "forward_inputs.pt")
        if override_model and (forward_args or forward_kwargs):
            forward_inputs = handle_nested_array(
                (forward_args, forward_kwargs), np.array)
            torch.save(forward_inputs, args_path)
        else:
            assert os.path.exists(args_path), "Missing forward args"
            forward_inputs = torch.load(args_path)
        self.forward_args, self.forward_kwargs = forward_inputs

        weights_path = os.path.join(root_path, "weights.pt")
        if isinstance(checkpoint, str):
            print(f"{self.name} loading checkpoint from {checkpoint}")
            weights = torch.load(checkpoint)
        elif not override_model: weights = None
        else: weights = get_torch_model_weights(raw_model)
        if weights: save_weights(weights, weights_path)

        ModelOnnxManager.__init__(self, self.name, self.torch_path)
        self.use_onnxs[self.name] = use_onnx
        asyncio.create_task(self.initialize_workers(
        self.name, num_workers, max_batch_size, local_mode))

    async def get_weights_publisher(self, name, node_id):
        if node_id == ray.get_runtime_context().get_node_id():
            return ray.get_runtime_context().current_actor
        pub = self.get_weights_publisher_meta(name)
        if node_id in self.weights_publishers:
            return self.weights_publishers[node_id]
        weights_publisher = ray.remote(ModelWeightsPublisher).options(
            num_cpus=0,
            scheduling_strategy=ray_scheduling_local(node_id),
            max_concurrency=100000,
        ).remote(self.name)
        self.weights_publishers[node_id] = weights_publisher
        return self.weights_publishers[node_id]

    async def set_weights(self, name, weights_ref, step):
        pub = self.get_weights_publisher_meta(name)
        # pub.weights, weights_ref = weights, ray.put(weights)
        pub.weights = weights = ray.get(weights_ref[0])
        if not pub.step: pub.step = self.get_ckpt_steps(name)[-1]
        step = pub.step + 1 if step == -1 else step
        publish_coros = [
            p.set_weights.remote(name, weights_ref[0], step)
            for p in self.weights_publishers.values()]
        await asyncio.gather(
            super().set_weights(name, weights, step), *publish_coros)
        merge(f"step/{name}", step, desc={
            "time_window_cnt": "step update per minute",
            "time_window_avg": "smoothed current step"})
        await self.on_set_weights(name, weights, step)

    async def clone(self, name, step, extra_name="", 
        max_batch_size=None, num_workers=0, 
        use_onnx=None, local_mode=None, gpus_per_worker=None
    ) -> str:
        step, weights = await self.get_target_step(name, step)
        cloned_name = f"{name}/clone-step-{step}"
        if extra_name: cloned_name = f"{cloned_name}-{extra_name}"
        if cloned_name in self.use_onnxs: return cloned_name
        if use_onnx is None: use_onnx = self.use_onnxs[name]
        self.use_onnxs[cloned_name] = use_onnx
        weights_path = os.path.join(self.trial_path, 
            f"{cloned_name}/weights.pt")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        if not os.path.exists(weights_path):
            await asyncio.get_running_loop().run_in_executor(
            None, save_weights, 
            weights or await self.get_weights(name, step), weights_path)
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        if max_batch_size is None: 
            max_batch_size = meta.max_batch_size
        if num_workers == -1: num_workers = meta.num_workers
        if local_mode is None: local_mode = meta.local_mode
        if gpus_per_worker is not None: 
            self.gpus_per_worker = gpus_per_worker
        await self.initialize_workers(
            cloned_name, num_workers, max_batch_size, local_mode)
        return cloned_name

    async def _clone(self, names: List[str]):
        base_name, parts = "/".join(names[:-1]), names[-1].split("-")
        if "/".join(names) in self.use_onnxs: return
        await self._clone(names[:-1])
        extra_name = "" if len(parts) < 4 else parts[3]
        await self.clone(base_name, int(parts[2]), extra_name)
        

    async def _is_health(self, worker):
        try: return await worker.forward.remote(self.forward_args, 
            self.forward_kwargs)
        except ray.exceptions.RayActorError: return False
        except Exception as e:
            return print(f"Worker is not health: ", e)

    async def get_target_step(self, name, step) -> Tuple[int, object]:
        pub = self.get_weights_publisher_meta(name)
        if step == -1 and pub.weights: return pub.step, pub.weights
        ckpt_steps = self.get_ckpt_steps(name)
        if step == -1: return ckpt_steps[-1], None
        index = np.searchsorted(ckpt_steps, step, side="right")
        return ckpt_steps[max(0, index - 1)], None

    async def get_weights(self, name, step=-1) -> NestedArray:
        step, weights = await self.get_target_step(name, step)
        if weights: return weights
        return await asyncio.get_running_loop().run_in_executor(
        None, self.load_checkpoint, name, step)

    async def get_model(self, name, use_onnx=None) -> Tuple:
        if use_onnx is None: use_onnx = self.use_onnxs[name]
        if use_onnx: return (use_onnx, 
            await self.get_onnx_model(name, use_onnx))
        if not self.model:
            self.model = ray.put(torch.load(self.torch_path))
        return use_onnx, self.model

    async def get_forward_inputs(self) -> Tuple[NestedArray]:
        return self.forward_args, self.forward_kwargs

    async def get_initialize_info(self, name) -> Tuple:
        if name not in self.w_metas: await self._clone(name.split("/"))
        meta: ModelWorkerManagerMeta = self.w_metas[name]
        if meta.pending_create_workers != 0: await asyncio.sleep(1)
        while meta.pending_create_workers != 0:
            await asyncio.sleep(3)
            print(f"Wait {name} worker to be initialized")
        return (meta.max_batch_size, self.cpus_per_worker, 
        self.gpus_per_worker, self.use_onnxs[name], meta.local_mode)


class RemoteModel:
    """RemoteModel封装了一个PyTorch模型，支持分布式部署和推理"""
    remote_models: Dict[str, "RemoteModel"] = {}
    max_cached_remote_model: int = 5

    def __new__(cls, name: str = None,
        model: torch.nn.Module = None,
        forward_args: Tuple[np.ndarray] = (),
        forward_kwargs: Dict[str, np.ndarray] = {},
        checkpoint_interval: int = None,
        checkpoint: Union[str, int] = None,
        max_batch_size: int = None,
        num_workers: int = None,
        cpus_per_worker: float = None,
        gpus_per_worker: float = None,
        memory_per_worker: int = None,
        use_onnx: ["train", "infer", "quantize"] = None,
        local_mode: [True, False, "proxy"] = None,
        port: int = None,
        override_model: bool = True,
    ):
        if name in cls.remote_models and (
            (self := cls.remote_models[name]) is not None
        ):
            return self
        self = super().__new__(cls)
        if name is None: return self    # 适配对象反序列化时调用__new__方法
        self.name, base_name = name, name.split("/clone")[0]
        self.cached_cloned_names = {}
        self.local_worker = None
        model_ = None
        if m := cls.remote_models.get(base_name, None):
            model_ = m.model
        self.model = model_ or Model.options(
            name=base_name,
            get_if_exists=True,
            scheduling_strategy=ray_scheduling_local(),
            max_concurrency=100000,
        ).remote(
            name, model,
            forward_args, forward_kwargs,
            checkpoint_interval, checkpoint,
            max_batch_size or 1,
            num_workers,
            cpus_per_worker or 1.0,
            gpus_per_worker or 0.0,
            memory_per_worker or 1024,
            use_onnx or "",
            local_mode or False,
            port,
            override_model,
        )
        (self.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, 
            self.use_onnx, l_mode
        ) = ray.get(self.model.get_initialize_info.remote(name))
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if cpus_per_worker is not None:
            self.cpus_per_worker = cpus_per_worker
        if gpus_per_worker is not None:
            self.gpus_per_worker = gpus_per_worker
        if local_mode is None: local_mode = l_mode
        if use_onnx is not None:
            self.use_onnx = use_onnx
        self.load_balance = WorkerLoadBalance(self.name, self.model)
        self._forward = self._forward_remote
        if local_mode is True: self._forward = self._forward_local
        if local_mode == "proxy":
            self._forward = self._forward_proxy
            self.forward_proxy = None
        cls.remote_models[name] = self
        names = list(cls.remote_models.keys())
        if len(names) > cls.max_cached_remote_model:
            cls.remote_models.pop(names[0])  # pop the oldest one
        return self

    def __init__(self, name: str, *args, **kwargs):
        """
        创建或者获取一个RemoteModel，如果已经存在同名的RemoteModel，则直接返回
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 
        目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            forward_args: 
        模型forward的位置参数输入，用于初始化模型
            forward_kwargs: 
        模型forward的关键字参数输入，用于初始化模型
            checkpoint_interval: 
        模型的checkpoint间隔，单位step，默认10分钟保存一次
            max_batch_size: 
        模型的max_batch_size，默认为1
            num_workers: 
        模型的worker数量，为None会自动根据负载情况调整，默认为0
            cpus_per_worker: 
        每个worker的CPU核心数，默认为1
            gpus_per_worker: 
        每个worker的GPU数量，如果为0，则表示不使用GPU，默认为0
            memory_per_worker: 
        每个worker的内存大小，单位MB，默认为1024
            use_onnx: 
        默认不适用onnx优化，可选值为["train", "infer"]
            local_mode: 
        为True模型会在本地运行，否则在Ray集群中运行，默认为False
            override_model: 
        为True会覆盖已经存在的模型，设为False可以加速启动
        """
        assert name in RemoteModel.remote_models, f"{name} not exist"

    def __del__(self):
        if self.local_worker: self.local_worker.is_initialized = False
        self.load_balance.is_initialized = False

    def _initialize_local_worker(self) -> ModelWorker:
        self.local_worker = ModelWorker(self.name, 
        self.max_batch_size, self.cpus_per_worker, self.gpus_per_worker, 
        use_onnx=self.use_onnx, model=self.model)

    def _initialize_forward_proxy(self) -> ModelForwardProxy:
        self.forward_proxy = ModelForwardProxy(self.max_batch_size, 4)
        load_balance = self.load_balance

        async def proxy_forward(args, kwargs):
            # TODO: add load_balance sync fault tolerance
            return await (await load_balance.select()
            ).forward.remote(args, kwargs)
        self.forward_proxy.proxy_forward = proxy_forward

    async def _forward_local(self, args, kwargs):
        if not self.local_worker: self._initialize_local_worker()
        return await self.local_worker.forward(args, kwargs)

    async def _forward_proxy(self, args, kwargs):
        if not self.forward_proxy: self._initialize_forward_proxy()
        return await self.forward_proxy.forward(args, kwargs)

    async def _forward_remote(self, args, kwargs, retry=2):
        worker = await self.load_balance.select()
        pending = len(self.load_balance.workers) < 2 or retry < 1
        try: return await worker.forward.remote(args, kwargs, pending)
        except ray.exceptions.RayActorError:
            print("Ray exception from model forward")
            await self.load_balance.sync()
        except ray.exceptions.RayTaskError:
            if (retry := retry - 1) < 0: raise
        return await self._forward_remote(args, kwargs, retry)

    async def forward(
        self, *args: NestedArray, batch=False, **kwargs: NestedArray
    ) -> NestedArray:
        """
        执行模型的前向计算，返回模型的输出，请注意batch维度的特殊处理
        Args:
            *args: 模型的位置参数输入，为一个或者多个 np.ndarray
            batch: 输入和输出是否包含batch维度，默认不包含
            **kwargs: 模型的关键字参数输入，为 np.ndarray 字典
        Returns:
            模型的输出，是一个或多个 np.ndarray
        """
        forward_beg, mode = time.time(), self._forward.__name__[9:]
        if not batch:
            args, kwargs = handle_nested_array((args, kwargs), 
                lambda x: np.expand_dims(x, 0))
        outputs = await self._forward(args, kwargs)
        if not batch:
            outputs = handle_nested_array(
                outputs, lambda x: np.squeeze(x, 0))
        merge_time_ms(f"forward/{self.name}", forward_beg, mode=mode)
        return outputs
    
    def __call__(self, *args, batch=True, **kwargs):
        """同 self.forward，内部封装了async调用"""
        from bray.utils import create_or_get_event_loop
        
        loop = create_or_get_event_loop()
        return asyncio.run_coroutine_threadsafe(
        self.forward(*args, batch=batch, **kwargs), loop).result()
    
    @property
    def step(self) -> int:
        """获取模型的最新版本号，每次调用 publish weights 会增加版本号"""
        return ray.get(self.model.get_step.remote(self.name))

    def set_local_mode(
        self, max_batch_size: int = 1, cpus_per_worker: float = 0.5, 
        gpus_per_worker: float = 0.0,
        use_onnx: ["train", "infer", "quantize"] = None,
    ):
        (
            self.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker
        ) = (
            max_batch_size, cpus_per_worker, gpus_per_worker
        )
        if use_onnx is not None: self.use_onnx = use_onnx
        self._forward = self._forward_local

    def get_model(self) -> torch.nn.Module:
        """
        获取被封装的原始模型，权重为最新权重，在Trainer里面会用到
        Returns:
            被封装的Pytorch模型，权重为最新的权重
        """
        torch_model = ray.get(ray.get(
            self.model.get_model.remote(self.name, use_onnx=""))[1])
        set_torch_model_weights(torch_model, 
            ray.get(self.model.get_weights.remote(self.name)))
        return torch_model

    def clone(self, step: int = -1, name=None, **kwargs):
        """
        克隆一个新的RemoteModel，可以用于SelfPlay和League的多智能体对抗
        Args:
            step: 克隆的模型的版本号，-1表示最新版本
            name: 克隆模型的额外名字后缀，用于区分不同模型
            kwargs: RemoteModel的关键字参数
        Returns:
            克隆的RemoteModel，用法和RemoteModel一致
        """
        # if cloned_name := self.cached_cloned_names.get(step, None):
        #     return RemoteModel(cloned_name, **kwargs)
        cloned_name = self.model.clone.remote(
            self.name, step, extra_name=name, **kwargs)
        cloned_name = ray.get(cloned_name)
        # if step != -1: self.cached_cloned_names[step] = cloned_name
        return type(self)(cloned_name, **kwargs)

    def publish_weights(self, weights: NestedArray, step=-1):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个NestedArray数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        weights = [ray.put(weights, _owner=self.model)]
        return self.model.set_weights.remote(self.name, weights, step)


class RemoteTorchModel(RemoteModel):
    def __call__(self, *args, batch=True, **kwargs) -> "NestedTensor":
        handle_input = lambda x: x.numpy() if isinstance(
            x, torch.Tensor) else x
        args, kwargs = handle_nested_array((args, kwargs), handle_input)
        outputs = super().__call__(*args, batch=batch, **kwargs)
        handle_output = lambda x: torch.as_tensor(x) if isinstance(
            x, np.ndarray) else x
        return handle_nested_array(outputs, handle_output)


if __name__ == "__main__":
    from bray.model.test import AtariModel, forward_args
    ray.init(namespace="model", address="local")

    model = AtariModel()

    remote_model = RemoteModel(
        "model", model=model, forward_args=forward_args)
    remote_model(*forward_args)

    remote_model = RemoteModel(
        "model1", model=model, forward_args=forward_args, 
        local_mode=True,
    )
    remote_model(*forward_args)

    remote_model = RemoteModel(
        "model2", model=model, forward_args=forward_args,
        port=9890,
    )
    print(remote_model(*forward_args))

    remote_model = RemoteModel(
        "model3", model=model, forward_args=forward_args,
        local_mode=True,
        use_onnx="infer",
    )
    remote_model(*forward_args)

    remote_model = RemoteModel(
        "model4", model=model, forward_args=forward_args,
        checkpoint_interval=1,
    )
    remote_model.clone()(*forward_args)
    remote_model.clone(name="1")(*forward_args)
    remote_model.clone().clone()(*forward_args)

    weights = get_torch_model_weights(model)
    ray.get(remote_model.clone().publish_weights(weights))
    ray.get(remote_model.publish_weights(weights))
    ray.get(remote_model.clone().clone().publish_weights(weights))

    # remote_model = RemoteModel(
    #     "model4", model=model, forward_args=forward_args,
    #     local_mode=True,
    #     use_onnx="quantize",
    # )
    # remote_model(*forward_args)
    print("All tests done")