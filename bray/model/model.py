from typing import List, Dict, Tuple, Union
import asyncio
import time
import random
import os
from concurrent.futures import ThreadPoolExecutor

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
)
from bray.utils.worker import WorkerLoadBalance
from bray.master.master import merge, query, merge_time_ms


def get_torch_model_weights(model: torch.nn.Module) -> Dict:
    return {k: v.cpu().detach().numpy() 
        for k, v in model.state_dict().items()}


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    model.load_state_dict(
        handle_nested_array(weights, torch.from_numpy))


def save_weights(weights: NestedArray, path: str):
    torch.save(handle_nested_array(weights, torch.as_tensor), path)


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
            args, kwargs = make_batch(
                pending_forwards, concate=True, parts=parts)
        try:
            outputs = await self.proxy_forward(args, kwargs)
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
        while (pending_size < self.max_batch_size and pending_size 
            != last_pending_size
        ):
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
        asyncio.create_task(
            self._forward_coro(self.forward_cond))

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
    def __init__(
        self, name, model: "Model", num_cpus, num_gpus, **kwargs
    ):
        self.name, self.model = name, model
        self.num_cpus, self.num_gpus = num_cpus, num_gpus

    def initialize_runtime(self):
        model = ray.get(ray.get(self.model.get_model.remote()))
        num_cpus = max(1, int(self.num_cpus))
        if torch.get_num_interop_threads() != num_cpus:
            torch.set_num_interop_threads(num_cpus)
        if torch.get_num_threads() != num_cpus: 
            torch.set_num_threads(num_cpus)
        model = model.requires_grad_(False).eval()
        weights = ray.get(self.model.get_weights.remote(self.name))
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

    def __init__(self, name, 
        model: "Model", num_cpus, num_gpus, use_onnx, **kwargs
    ):
        self.name, self.model = name, model
        self.num_cpus, self.num_gpus = num_cpus, num_gpus
        self.use_onnx = use_onnx

    def build_ort_session(self):
        onnx_model, forward_outputs = ray.get(
            self.model.get_onnx_model.remote(self.name, self.use_onnx))
        forward_outputs = handle_nested_array(
            forward_outputs, lambda x: x.copy())
        provider = ["CUDAExecutionProvider"] if self.num_gpus else []
        num_cpus = max(1, int(self.num_cpus))
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_cpus
        options.inter_op_num_threads = num_cpus
        return ort.InferenceSession(
            onnx_model, options, provider), forward_outputs

    def get_cached_onnx_session(self):
        if self.use_onnx != "train": return None, None
        name = self.name.split("/")[0]
        if name not in OnnxModelWorker.cached_onnx_session:
            return None, None
        return OnnxModelWorker.cached_onnx_session[name]

    def set_cached_onnx_session(self, sess, outputs):
        if self.use_onnx != "train": return
        name = self.name.split("/")[0]
        OnnxModelWorker.cached_onnx_session[name] = (sess, outputs)

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
        self.weights = [weights[n].copy() 
            for n in self.input_names if n in weights]


class ModelWorker(ModelForwardProxy):
    executor: ThreadPoolExecutor = None

    def __init__(self, name, step, max_batch_size, cpus_per_worker, 
        gpus_per_worker, use_onnx, model: "Model" = None
    ):
        super().__init__(max_batch_size, parallel=1)

        self.name, self.model = name, model or ray.get_actor(
            name.split("/")[0])
        (
            self.current_step, self.max_batch_size,
            self.num_cpus, self.num_gpus, self.use_onnx
        ) = (
            step, max_batch_size, 
            cpus_per_worker, gpus_per_worker, use_onnx
        )
            
        if self.use_onnx: RuntimeWorker = OnnxModelWorker
        else: RuntimeWorker = TorchModelWorker

        self.runtime_worker = RuntimeWorker(
            name=self.name, model=self.model, 
            num_cpus=self.num_cpus, 
            num_gpus=self.num_gpus, use_onnx=self.use_onnx)

        self.runtime_worker.initialize_runtime()

        if not ModelWorker.executor:
            ModelWorker.executor = ThreadPoolExecutor(max_workers=1)

        self.is_initialized = False

    async def proxy_forward(self, args, kwargs):
        forward_beg = time.time()
        outputs = await asyncio.get_running_loop().run_in_executor(
            ModelWorker.executor, 
            self.runtime_worker.forward, args, kwargs)
        merge_time_ms(f"forward/{self.name}", forward_beg)
        return outputs

    async def forward(self, args, kwargs, pending=True):
        if not self.is_initialized:
            self.is_initialized = True
            asyncio.create_task(self.subscribe_weights())
        self.forward = super().forward
        return await self.forward(args, kwargs, pending)

    async def _subscribe_weights(self):
        subscribe_beg = time.time()
        weights, step = await self.model.subscribe_weights.remote(
            self.name, self.current_step
        )
        if step == self.current_step: return
        weights = await asyncio.wait_for(weights, timeout=1)
        await asyncio.get_running_loop().run_in_executor(
            ModelWorker.executor, 
            self.runtime_worker.set_weights, weights)
        self.current_step = step
        merge_time_ms(f"subscribe/{self.name}", subscribe_beg)

    async def subscribe_weights(self):
        if len(self.name.split("/")) != 1 or not self.is_initialized:
            return
        try: await self._subscribe_weights()
        except Exception as e:
            await asyncio.sleep(0.1)
            print(f"Fail to subscribe weights for {self.name}: {e}")
        asyncio.create_task(self.subscribe_weights())

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


class ModelMeta:
    def __init__(
        self, num_workers=None, use_onnx=None, local_mode=False
    ):
        self.num_workers, self.workers = num_workers, []
        self.max_batch_size = 1
        self.weights, self.step, self.ckpt_step = None, 0, 0
        self.use_onnx, self.onnx_step = use_onnx, -1
        self.local_mode = local_mode
        self.pending_create_workers = 0
        self.ckpt_steps = []
        self.step_cond = asyncio.Condition()
        self.last_sub_weights, self.last_sub_step = None, -1
        self.worker_cond = asyncio.Condition()


@ray.remote(num_cpus=0)
class Model:
    def __init__(
        self,
        name: str,
        torch_model: torch.nn.Module = None,
        forward_args: Tuple[np.ndarray] = None,
        forward_kwargs: Dict[str, np.ndarray] = None,
        checkpoint_interval: int = None,
        checkpoint: Union[str, int] = None,
        max_batch_size: int = 1,
        num_workers: int = None,
        cpus_per_worker: float = 1.0,
        gpus_per_worker: float = 0.0,
        memory_per_worker: int = 1024,
        use_onnx: str = None,
        local_mode: bool = False,
        override_model: bool = True,
    ):
        self.trial_path = ray.get_runtime_context().namespace
        root_path = os.path.join(self.trial_path, f"{name}")
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)

        self.torch_path = os.path.join(root_path, f"{name}.pt")
        if torch_model is not None:
            torch.save(torch_model, self.torch_path)
        else:
            assert os.path.exists(
                self.torch_path
            ), f"Model {name} not found, missing torch model"
        self.name, self.model = name, None
        self.override_model = torch_model is not None and override_model

        args_path = os.path.join(root_path, "forward_inputs.pt")
        if forward_args or forward_kwargs:
            forward_args, forward_kwargs = handle_nested_array(
                (forward_args, forward_kwargs), np.array
            )
            torch.save((forward_args, forward_kwargs), args_path)
        else:
            assert os.path.exists(args_path), "Missing forward args"
            forward_args, forward_kwargs = torch.load(args_path)
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs

        self.checkpoint_interval = checkpoint_interval
        self.models: Dict[str, ModelMeta] = {}

        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1))

        weights_path = os.path.join(self.trial_path, f"{name}/weights.pt")

        if isinstance(checkpoint, str):
            print(f"Model {name} loading checkpoint from {checkpoint}")
            weights = torch.load(checkpoint)
        elif torch_model is None or (
            not self.override_model and os.path.exists(weights_path)):
            weights = None
        else: weights = get_torch_model_weights(torch_model)

        if weights: save_weights(weights, weights_path)

        meta = ModelMeta(num_workers, use_onnx, local_mode)
        self._initialize_model(self.name, max_batch_size, meta, checkpoint)

        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.memory_per_worker = memory_per_worker

        self.RemoteModelWorker = ray.remote(ModelWorker).options(
            num_cpus=cpus_per_worker,
            num_gpus=gpus_per_worker,
            memory=memory_per_worker * 1024 * 1024,
            scheduling_strategy="SPREAD",
        )
        for _ in range(
            len([node for node in ray.nodes() if node["Alive"]])
            if num_workers is None
            else num_workers
        ):
            asyncio.create_task(self._create_worker(self.name))

        asyncio.create_task(self._health_check())
        if self.checkpoint_interval is None:
            asyncio.create_task(self._save_checkpoint())

    def _build_ckpt_steps(self, name, ckpt_dir, checkpoint):
        if isinstance(checkpoint, str) or checkpoint and checkpoint <= 0:
            return []
        clone_steps = [
            int(ckpt.split(".")[0].split("-")[2])
            for ckpt in os.listdir(os.path.join(self.trial_path, f"{name}"))
            if ckpt.startswith("clone-step")
        ]
        ckpt_steps = [
            int(ckpt.split(".")[0].split("-")[1]) for ckpt in os.listdir(ckpt_dir)
        ]
        # union of ckpt_steps and clone_steps
        ckpt_steps = list(set(ckpt_steps).union(clone_steps))
        ckpt_steps.sort()
        is_valid = lambda ckpt: checkpoint is None or ckpt < checkpoint
        return [ckpt for ckpt in ckpt_steps if is_valid(ckpt)]

    def _initialize_model(self, name, max_batch_size, meta, checkpoint=None):
        ckpt_dir = os.path.join(self.trial_path, f"{name}/checkpoint")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        try:
            ckpt_steps = self._build_ckpt_steps(name, ckpt_dir, checkpoint)
            step = ckpt_steps[-1] if ckpt_steps else 0
            (
                meta.step, meta.ckpt_step, meta.ckpt_steps
            ) = step, step, ckpt_steps
        except Exception as e:
            print(f"Build checkpoint steps for {name} failed: {e}")

        if meta.step != 0:
            print(f"Model {name} restore checkpoint step is {meta.step}")

        meta.max_batch_size = max_batch_size
        self.models[name] = meta

        if meta.use_onnx:
            onnx_model, forward_outputs = self._get_onnx_model(
                name, meta.use_onnx)
        else:
            onnx_model, forward_outputs = None, None
        if meta.use_onnx and name == self.name:
            print(f"Using onnx model for {name} {meta.use_onnx}.")
        if name != self.name:
            return
        self.onnx_model, self.forward_outputs = onnx_model, forward_outputs

    def _get_onnx_model(self, name, use_onnx) -> Tuple[bytes, NestedArray]:
        onnx_path_postfix = f"{name}/{name}.onnx"
        onnx_path = os.path.join(self.trial_path, onnx_path_postfix)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        outputs_path = os.path.join(
            self.trial_path,
            f"{self.name}/forward_outputs.pt",
        )
        from bray.model.onnx import export_onnx

        meta: ModelMeta = self.models[name]
        step, weights = self._get_target_step(name, -1)
        onnx_step = meta.onnx_step
        if (use_onnx != "train" and (
            (self.override_model and onnx_step == -1)
            or (onnx_step >= 0 and onnx_step < step)
            or not os.path.exists(onnx_path)
            or not os.path.exists(outputs_path))
        ):
            weights = ray.get(weights) if weights else self._load_checkpoint(
                name, step)
            torch_model = torch.load(self.torch_path)
            set_torch_model_weights(torch_model, weights)
            print(f"Exporting latest onnx model at step {step} to", onnx_path)
            forward_outputs = export_onnx(
                torch_model,
                onnx_path,
                self.forward_args,
                self.forward_kwargs,
                export_params=True,
                quantize=use_onnx == "quantize",
            )
            meta.onnx_step = step
            torch.save(forward_outputs, outputs_path)

        onnx_train_path = os.path.join(self.trial_path, f"{self.name}/train.onnx")
        meta: ModelMeta = self.models[self.name]
        if use_onnx != "train" or (
            (not self.override_model or meta.onnx_step != -1)
            and os.path.exists(onnx_train_path)
            and os.path.exists(outputs_path)
        ):
            if use_onnx == "train":
                onnx_path = onnx_train_path
            with open(onnx_path, "rb") as f:
                onnx_model = f.read()
            return onnx_model, torch.load(outputs_path)

        print("Exporting onnx model to", onnx_train_path)
        torch_model = torch.load(self.torch_path)
        forward_outputs = export_onnx(
            torch_model,
            onnx_train_path,
            self.forward_args,
            self.forward_kwargs,
            export_params=False,
        )
        meta.onnx_step = 0
        torch.save(forward_outputs, outputs_path)
        with open(onnx_train_path, "rb") as f:
            onnx_model = f.read()
        return onnx_model, forward_outputs

    async def _create_worker(self, name):
        meta: ModelMeta = self.models[name]
        if meta.pending_create_workers > (meta.num_workers or 10):
            return
        worker = self.RemoteModelWorker.remote(
            name, meta.step, meta.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, meta.use_onnx)
        meta.pending_create_workers += 1
        if not await self._is_health(worker):
            meta.pending_create_workers -= 1
            return
        meta.pending_create_workers -= 1
        if (meta.num_workers is not None 
            and len(meta.workers) >= meta.num_workers):
            return
        worker.__bray_node_id = await worker.get_node_id.remote()
        meta.workers.append(worker)
        async with meta.worker_cond:
            meta.worker_cond.notify_all()

    async def set_weights(self, name, weights, step):
        meta: ModelMeta = self.models[name]
        meta.weights = weights[0]
        meta.step = meta.step + 1 if step == -1 else step
        merge(
            f"step/{name}", meta.step,
            desc={
                "time_window_cnt": "step update per minute",
                "time_window_avg": "smoothed current step",
            })
        async with meta.step_cond:
            meta.step_cond.notify_all()
        if (
            self.checkpoint_interval and (meta.step // self.checkpoint_interval > 
            meta.ckpt_step // self.checkpoint_interval)
        ):
            meta.ckpt_step = meta.step
            await self.__save_checkpoint(
                name, meta.ckpt_step, meta.weights)

    async def clone(self, name, step, 
        max_batch_size, num_workers, use_onnx, local_mode, **kwargs
    ) -> str:
        step, weights = self._get_target_step(name, step)

        if (cloned_name := f"{name}/clone-step-{step}") in self.models:
            return cloned_name

        weights = await (weights if weights else self.get_weights(name, step))
        meta: ModelMeta = self.models[name]
        use_onnx = use_onnx if use_onnx != "" else meta.use_onnx
        max_batch_size = (
            max_batch_size 
            if max_batch_size is not None else meta.max_batch_size
        )
        num_workers = num_workers if num_workers != -1 else meta.num_workers

        loop = asyncio.get_running_loop()
        cloned_meta = ModelMeta(num_workers, use_onnx, local_mode)

        def initialize_model_if_needed():
            if cloned_name in self.models: return

            weights_path = os.path.join(
                self.trial_path, f"{cloned_name}/weights.pt",
            )
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            if not os.path.exists(weights_path):
                save_weights(weights, weights_path)

            self._initialize_model(
                cloned_name, max_batch_size, cloned_meta
            )
            for _ in range(
                len([node for node in ray.nodes() if node["Alive"]])
                if num_workers is None
                else num_workers
            ):
                loop.create_task(self._create_worker(cloned_name))

        await loop.run_in_executor(None, initialize_model_if_needed)
        return cloned_name

    async def subscribe_weights(self, name, current_step):
        meta: ModelMeta = self.models[name]
        async with meta.step_cond:
            pred = lambda: meta.step != current_step
            try:
                await asyncio.wait_for(
                    meta.step_cond.wait_for(pred), timeout=10 * 60)
            except asyncio.TimeoutError: 
                return None, meta.step
        # if  meta.last_sub_step != current_step:
        #     return meta.last_sub_weights, meta.last_sub_step
        meta.cached_weights = meta.last_sub_weights
        step = meta.last_sub_step = meta.step
        weights = meta.last_sub_weights = meta.weights
        return weights, step

    async def subscribe_workers(self, name, node_id=None, cur_num=0):
        meta: ModelMeta = self.models[name]
        async with meta.worker_cond:
            try:
                pred = lambda: cur_num != len(self.workers)
                await asyncio.wait_for(
                    meta.worker_cond.wait_for(pred), timeout=10 * 60)
            except asyncio.TimeoutError: pass
        return await self.get_workers(name, node_id)

    def _load_balance(self, name):
        meta: ModelMeta = self.models[name]
        if len(meta.workers) == 0:
            asyncio.create_task(self._create_worker(name))
            return
        forward_time_sum = query("forward", kind="sum", model=name)
        local_forward_time_sum = query(
            "forward", kind="sum", model=name, mode="local",
        )
        forward_time_sum -= local_forward_time_sum
        load_rate = max(0.0, forward_time_sum) / (len(meta.workers) * 60 * 1000)
        merge(
            f"load/{name}",
            load_rate,
            desc={"time_window_avg": "load rate of model forward"},
        )
        # 假设以概率p下掉worker，那么下掉后的worker数量为(1-p)*worker_num
        # 目标负载率为0.6，那么下掉后的负载量为(1-p)*worker_num*0.6
        # 它应该等于当前测得的负载量，
        # 即(1-p)*worker_num*0.6 == worker_num * load_rate
        # 解得p = 1 - load_rate / 0.6
        # 为了避免过度下掉，我们加入平滑因子 random.random() ** 2
        if load_rate < 0.4 and len(meta.workers) > 1:
            p = 1 - load_rate / 0.6
            shrink_num = min(2, int(p * random.random() ** 2 * len(meta.workers)))
            del meta.workers[len(meta.workers) - shrink_num :]
            return
        if load_rate < 0.55:
            return
        # 三级调控，保证负载均衡响应速度，同时避免过度调控
        add_rate = 0.5 if load_rate < 0.7 else (1 if load_rate < 0.8 else 1.5)
        for _ in range(1 + int(add_rate * len(meta.workers))):
            asyncio.create_task(self._create_worker(name))

    async def _is_health(self, worker):
        try:
            await worker.forward.remote(self.forward_args, self.forward_kwargs)
            return True
        except ray.exceptions.RayActorError:
            return False
        except Exception as e:
            print(f"Worker is not health: ", e)
            return False

    async def __health_check(self, name):
        meta: ModelMeta = self.models[name]
        origin_workers_num = len(meta.workers)
        active_workers = [
            worker for worker in meta.workers[:origin_workers_num]
            if await self._is_health(worker)
        ]
        old_workers = meta.workers
        meta.workers = active_workers
        meta.workers.extend(old_workers[origin_workers_num:])
        merge(
            f"worker/{name}",
            len(meta.workers),
            desc={"time_window_avg": "smoothed model worker num"},
        )
        if meta.num_workers is None:
            self._load_balance(name)
        elif len(meta.workers) < meta.num_workers:
            asyncio.create_task(self._create_worker(name))

    async def _health_check(self):
        await asyncio.sleep(60)  # wait for workers to start
        for name in list(self.models.keys()):
            await self.__health_check(name)
        asyncio.create_task(self._health_check())

    def _load_checkpoint(self, name, step):
        weights_path = os.path.join(
            os.path.join(
                self.trial_path, f"{name}/checkpoint"), f"step-{step}.pt")
        if not os.path.exists(weights_path) and step == 0:
            weights_path = os.path.join(
                self.trial_path, f"{name}/weights.pt")
        if not os.path.exists(weights_path):
            clone_dir = os.path.join(
                self.trial_path, f"{name}/clone-step-{step}")
            weights_path = os.path.join(clone_dir, f"weights.pt")
        return handle_nested_array(
            torch.load(weights_path), 
            lambda x: x.numpy(), type_check=False
        )

    async def __save_checkpoint(self, name, ckpt_step, weights):
        meta: ModelMeta = self.models[name]
        ckpt_dir = os.path.join(self.trial_path, f"{name}/checkpoint")
        ckpt_path = os.path.join(
            ckpt_dir,
            f"step-{ckpt_step}.pt",
        )
        await asyncio.get_running_loop().run_in_executor(
            None, torch.save, 
            handle_nested_array(await weights, torch.from_numpy), ckpt_path)
        meta.ckpt_steps.append(ckpt_step)

    async def _save_checkpoint(self):
        await asyncio.sleep(10 * 60)  # save checkpoint every 10 minutes
        for name, meta in list(self.models.items()):
            if meta.ckpt_step >= meta.step:
                continue
            meta.ckpt_step = meta.step
            await self.__save_checkpoint(name, meta.ckpt_step, meta.weights)
        asyncio.create_task(self._save_checkpoint())

    def _get_target_step(self, name, step) -> Tuple[int, object]:
        meta: ModelMeta = self.models[name]
        if step == -1 and meta.weights:
            return meta.step, meta.weights
        ckpt_steps = [0] + meta.ckpt_steps
        if step == -1:
            return ckpt_steps[-1], None
        index = np.searchsorted(ckpt_steps, step, side="right")
        return ckpt_steps[max(0, index - 1)], None

    async def get_weights(self, name, step=-1) -> NestedArray:
        step, weights = self._get_target_step(name, step)
        if weights:
            return await weights
        return await asyncio.get_running_loop().run_in_executor(
            None, self._load_checkpoint, name, step)

    async def get_step(self, name) -> int:
        return self.models[name].step

    async def get_ckpt_steps(self, name) -> List[int]:
        return self.models[name].ckpt_steps

    async def get_model(self) -> ray.ObjectRef:
        if not self.model:
            self.model = ray.put(torch.load(self.torch_path))
        return self.model

    async def get_forward_inputs(self) -> Tuple[NestedArray]:
        return self.forward_args, self.forward_kwargs

    async def get_onnx_model(
        self, name, use_onnx=None) -> Tuple[bytes, NestedArray]:
        meta: ModelMeta = self.models[name]
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_onnx_model, name, use_onnx or meta.use_onnx
        )

    async def get_workers(
        self, name, node_id: str = None) -> List[ModelWorker]:
        model: ModelMeta = self.models[name]
        if not node_id:
            return model.workers
        workers = [w for w in model.workers if w.__bray_node_id == node_id]
        return workers if workers else model.workers

    async def get_initialize_info(self, name) -> Tuple:
        meta: ModelMeta = self.models[name]
        interval = 1
        while meta.pending_create_workers != 0:
            print(f"Wait {name} worker to be initialized")
            await asyncio.sleep(interval)
            interval = min(30, interval * 2)
        return (
            meta.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, meta.use_onnx, meta.local_mode
        )


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个ModelWorker实现并行计算
    """

    remote_models: Dict[str, "RemoteModel"] = {}
    max_cached_remote_model: int = 5

    def __new__(
        cls,
        name: str = None,
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
        override_model: bool = False,
    ):
        if name in cls.remote_models and (
            (self := cls.remote_models[name]) is not None
        ):
            return self
        self = super().__new__(cls)
        if name is None: return self    # 适配对象反序列化时调用__new__方法
        self.name, base_name = name, name.split("/")[0]
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
            base_name,
            model,
            forward_args,
            forward_kwargs,
            checkpoint_interval,
            checkpoint,
            max_batch_size or 1,
            num_workers,
            cpus_per_worker or 1.0,
            gpus_per_worker or 0.0,
            memory_per_worker or 1024,
            use_onnx,
            local_mode or False,
            override_model,
        )
        (
            self.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, 
            self.use_onnx, l_mode
        ) = ray.get(self.model.get_initialize_info.remote(name))
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
        if cpus_per_worker is not None:
            self.cpus_per_worker = cpus_per_worker
        if gpus_per_worker is not None:
            self.gpus_per_worker = gpus_per_worker
        local_mode = l_mode if local_mode is None else local_mode
        if use_onnx is not None:
            self.use_onnx = use_onnx
        self.worker_load_balance = WorkerLoadBalance(self.name, self.model)
        self._forward = self._remote_forward
        if local_mode is True: self._forward = self._local_forward
        if local_mode == "proxy":
            self._forward = self._proxy_forward
            self.model_forward_proxy = None
        cls.remote_models[name] = self
        names = list(cls.remote_models.keys())
        if len(names) > cls.max_cached_remote_model:
            cls.remote_models.pop(names[0])  # pop the oldest one
        return self

    def __init__(self, name: str, *args, **kwargs):
        """
        创建或者获取一个RemoteModel，如果已经存在同名的RemoteModel，则直接返回
        Args:
            name: 
        模型的名字，用于在Ray集群中标识模型
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
        assert name in RemoteModel.remote_models, f"RemoteModel {name} not exist"

    def __del__(self):
        self.worker_load_balance.is_initialized = False
        if self.local_worker: self.local_worker.is_initialized = False

    def _new_local_worker(self) -> ModelWorker:
        return ModelWorker(
            self.name, self.step, self.max_batch_size, self.cpus_per_worker, 
            self.gpus_per_worker, self.use_onnx, self.model)

    def _new_forward_proxy(self) -> ModelForwardProxy:
        model_forward_proxy = ModelForwardProxy(self.max_batch_size, 4)
        worker_load_balance = self.worker_load_balance

        async def proxy_forward(args, kwargs):
            worker = await worker_load_balance.select()
            return await worker.forward.remote(args, kwargs)
        model_forward_proxy.proxy_forward = proxy_forward
        return model_forward_proxy

    async def _local_forward(self, args, kwargs) -> NestedArray:
        if not self.local_worker:
            self.local_worker = self._new_local_worker()
        forward_beg = time.time()
        outputs = await self.local_worker.forward(args, kwargs)
        merge_time_ms(f"forward/{self.name}", forward_beg, mode="local")
        return outputs

    async def _proxy_forward(self, args, kwargs) -> NestedArray:
        if not self.model_forward_proxy:
            self.model_forward_proxy = self._new_forward_proxy()
        forward_beg = time.time()
        outputs = await self.model_forward_proxy.forward(args, kwargs)
        merge_time_ms(f"forward/{self.name}", forward_beg, mode="proxy")
        return outputs

    async def _remote_forward(self, args, kwargs, retry=2) -> NestedArray:
        worker = await self.worker_load_balance.select()
        forward_beg = time.time()
        pending = len(self.worker_load_balance.workers) < 2 or retry < 1
        try:
            outputs = await worker.forward.remote(
                args, kwargs, pending)
        except ray.exceptions.RayActorError:
            print("Ray exception from model forward")
            return await self._remote_forward(args, kwargs)
        except ray.exceptions.RayTaskError:
            if (retry := retry - 1) < 0: raise
            return await self._remote_forward(args, kwargs, retry)
        merge_time_ms(f"forward/{self.name}", forward_beg, mode="remote")
        return outputs

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
        if not batch:
            args, kwargs = handle_nested_array(
                (args, kwargs), lambda x: np.expand_dims(x, 0))
        outputs = await self._forward(args, kwargs)
        if not batch:
            outputs = handle_nested_array(outputs, lambda x: np.squeeze(x, 0))
        return outputs
    
    def __call__(
        self, *args: NestedArray, batch=True, **kwargs: NestedArray
    ) -> NestedArray:
        """同 self.forward，内部封装了async调用"""
        from bray.utils import create_or_get_event_loop
        
        loop = create_or_get_event_loop()
        return asyncio.run_coroutine_threadsafe(self.forward(
            *args, batch=batch, **kwargs), loop).result()
    
    @property
    def step(self) -> int:
        """获取模型的最新版本号，每次调用 publish weights 会增加版本号"""
        if self.local_worker and self.local_worker.is_initialized:
            return self.local_worker.get_model_step()
        return ray.get(self.model.get_step.remote(self.name))
        # if not self.workers:
        #     return ray.get(self.model.get_step.remote(self.name))
        # worker = self.workers[random.randint(0, len(self.workers) - 1)]
        # return ray.get(worker.get_model_step.remote())

    def set_local_mode(
        self, max_batch_size: int = 1, cpus_per_worker: float = 0.5, 
        gpus_per_worker: float = 0.0,
        use_onnx: ["train", "infer", "quantize"] = None,
    ):
        (
            self.max_batch_size, 
            self.cpus_per_worker, self.gpus_per_worker
        ) = (
            max_batch_size, cpus_per_worker, gpus_per_worker
        )
        if use_onnx is not None:
            self.use_onnx = use_onnx
        self._forward = self._local_forward

    def get_model(self) -> torch.nn.Module:
        """
        获取被封装的原始模型，权重为最新权重，在Trainer里面会用到
        Returns:
            被封装的Pytorch模型，权重为最新的权重
        """
        torch_model = ray.get(ray.get(self.model.get_model.remote()))
        set_torch_model_weights(torch_model, 
            ray.get(self.model.get_weights.remote(self.name)))
        return torch_model

    def clone(self, step: int = -1, **kwargs) -> "RemoteModel":
        """
        克隆一个新的RemoteModel，可以用于SelfPlay和League的多智能体对抗
        Args:
            step: 克隆的模型的版本号，-1表示最新版本
            kwargs: RemoteModel的关键字参数
        Returns:
            克隆的RemoteModel，用法和RemoteModel一致
        """
        if name := self.cached_cloned_names.get(step, None):
            return RemoteModel(name, **kwargs)
        cloned_name = self.model.clone.remote(self.name, step, **kwargs)
        cloned_name = ray.get(cloned_name)
        if step != -1: self.cached_cloned_names[step] = cloned_name
        return RemoteModel(cloned_name, **kwargs)

    def publish_weights(self, weights: NestedArray, step=-1):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个NestedArray数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        # self.model.set_weights.remote(self.name, weights, step)
        weights = [ray.put(weights, _owner=self.model)]
        self.model.set_weights.remote(self.name, weights, step)

    def warmup(self):
        """预热模型，避免第一次forward的时候耗时过长"""
        if self._forward != self._local_forward or self.local_worker:
            return
        self.local_worker = ModelWorker(self.name, self.model)

    def get_torch_forward_args(self):
        return self.get_torch_forward_inputs()[0]

    def get_torch_forward_inputs(self):
        forward_args, forward_kwargs = ray.get(self.model.get_forward_inputs.remote())
        torch_args, torch_kwargs = handle_nested_array(
            (forward_args, forward_kwargs), torch.as_tensor
        )
        return torch_args, torch_kwargs
