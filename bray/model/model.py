import asyncio
import time
import random
import os
from concurrent.futures import ThreadPoolExecutor

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
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
from bray.metric.metric import merge, query, merge_time_ms
from bray.actor.actor import get_tick_id


def get_torch_model_weights(model: torch.nn.Module) -> NestedArray:
    return {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    model.load_state_dict(handle_nested_array(weights, torch.from_numpy))


class ModelWorker:
    ort_session_and_forward_outputs: dict[str, tuple] = {}

    def __init__(self, name: str, loop: asyncio.AbstractEventLoop = None):
        self.name, self.model = name, ray.get_actor(name.split("/")[0])
        self.current_step = ray.get(self.model.get_step.remote(name))

        self.max_batch_size = ray.get(self.model.get_max_batch_size.remote(name))
        self.pending_forwards: list[tuple, dict] = []
        self.ready_forwards: list[NestedArray] = []
        self.forward_cond = asyncio.Condition()

        self.use_onnx = ray.get(self.model.get_use_onnx.remote(name))
        self.num_cpus, self.num_gpus = ray.get(
            self.model.get_cpus_gpus_per_worker.remote()
        )
        self._init_onnx() if self.use_onnx else self._init_torch()
        self.__forward = self._forward_onnx if self.use_onnx else self._forward_torch

        loop = loop if loop else asyncio.get_running_loop()
        self.forward_task = loop.create_task(self._forward_coro())
        self.forward_task.add_done_callback(
            lambda t: None if t.cancelled() else t.result()
        )
        self.subscribe_task = loop.create_task(self._subscribe_weights())
        self.subscribe_task.add_done_callback(
            lambda t: None if t.cancelled() else t.result()
        )

    def _build_ort_session(self, onnx_model):
        provider = "CUDAExecutionProvider" if self.num_gpus else "CPUExecutionProvider"
        num_cpus = max(1, int(self.num_cpus))
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_cpus
        options.inter_op_num_threads = num_cpus
        return ort.InferenceSession(onnx_model, options, [provider])

    def _init_onnx(self):
        self.ort_session, self.forward_outputs = None, None
        base_name = self.name.split("/")[0]
        if self.use_onnx == "train":
            (
                self.ort_session,
                self.forward_outputs,
            ) = ModelWorker.ort_session_and_forward_outputs.get(base_name, (None, None))
        if not self.ort_session:
            onnx_model, self.forward_outputs = ray.get(
                self.model.get_onnx_model.remote(self.name)
            )
            self.ort_session = self._build_ort_session(onnx_model)
        if self.use_onnx != "train":
            return
        ModelWorker.ort_session_and_forward_outputs[base_name] = (
            self.ort_session,
            self.forward_outputs,
        )
        weights = ray.get(self.model.get_weights.remote(self.name))
        self.weights = [v for v in weights.values()]

    def _init_torch(self):
        num_cpus = max(1, int(self.num_cpus))
        torch.set_num_interop_threads(num_cpus)
        torch.set_num_threads(num_cpus)
        model = ray.get(ray.get(self.model.get_model.remote()))
        model.requires_grad_(False)
        model.eval()
        weights = ray.get(self.model.get_weights.remote(self.name))
        set_torch_model_weights(model, weights)
        if self.num_gpus:
            model = model.cuda()
        self.torch_model = model

    def _forward_torch(self, batch_args, batch_kwargs):
        handler = (
            (lambda x: torch.from_numpy(x).cuda())
            if self.num_gpus
            else torch.from_numpy
        )
        batch_args, batch_kwargs = handle_nested_array(
            (batch_args, batch_kwargs), handler
        )
        outputs = self.torch_model(*batch_args, **batch_kwargs)
        return handle_nested_array(
            outputs, lambda x: x.cpu().detach().numpy(), type_check=False
        )

    def _forward_onnx(self, batch_args, batch_kwargs):
        sess = self.ort_session
        input_names = [input.name for input in sess.get_inputs()]
        flatten_input = flatten_nested_array(
            batch_args + (batch_kwargs,), sort_keys=True
        )
        if self.use_onnx == "train":
            flatten_input.extend(self.weights)
        inputs = dict(zip(input_names, flatten_input))
        # output_names = [output.name for output in sess.get_outputs()]
        # print(handle_nested_array(inputs, lambda x: (x.shape, x.dtype)))
        outputs = sess.run(None, inputs)
        return unflatten_nested_array(self.forward_outputs, outputs)

    async def _forward(self, pending_forwards: list[tuple, dict]):
        beg = time.time()
        batch_args, batch_kwargs = make_batch(pending_forwards)
        outputs = await asyncio.get_running_loop().run_in_executor(
            None, self.__forward, batch_args, batch_kwargs
        )
        ready_forwards = split_batch(outputs)
        merge_time_ms("forward", beg, model=self.name)
        return ready_forwards

    async def _forward_coro(self, forward_cond=asyncio.Condition()):
        while True:
            pending_forwards = self.pending_forwards
            async with self.forward_cond:
                await self.forward_cond.wait_for(lambda: pending_forwards)
            self.pending_forwards = []
            ready_forwards = self.ready_forwards
            self.ready_forwards = []
            forward_cond, self.forward_cond = self.forward_cond, forward_cond
            # set the ready_forwards to the previous pending_forwards
            ready_forwards[:] = await self._forward(pending_forwards)
            async with forward_cond:
                forward_cond.notify_all()

    async def forward(self, *args, **kwargs) -> NestedArray:
        if self.max_batch_size == 1:
            return (await self._forward([(args, kwargs)]))[0]
        while len(self.pending_forwards) >= self.max_batch_size:
            await asyncio.sleep(0.001)
        forward_index = len(self.pending_forwards)
        self.pending_forwards.append((args, kwargs))
        previous_forward_cond = self.forward_cond
        previous_ready_forwards = self.ready_forwards
        async with previous_forward_cond:
            if forward_index == 0:
                previous_forward_cond.notify_all()
            await previous_forward_cond.wait_for(lambda: previous_ready_forwards)
        return previous_ready_forwards[forward_index]

    async def __subscribe_weights(self):
        try:
            weights, step = await self.model.subscribe_weights.remote(
                self.name, self.current_step
            )
        except Exception as e:
            print(f"Fail to subscribe weights from {self.name}.", e)
        if step <= self.current_step:
            print(f"Skip weights from {self.name}.")
            return
        self.current_step = step
        if not self.use_onnx:
            set_torch_model_weights(self.torch_model, await weights)
        elif self.use_onnx == "train":
            self.weights = [v for v in (await weights).values()]
        else:
            print("Set onnx weights only in train mode.")

    async def _subscribe_weights(self):
        while True:
            await self.__subscribe_weights()

    def get_model_step(self) -> int:
        """Get the current step of the model from worker to reduce Model overhead."""
        return self.current_step

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


class ModelMeta:
    def __init__(self, num_workers: int = None, use_onnx: str = None):
        self.num_workers, self.workers = num_workers, []
        self.max_batch_size = 1
        self.weights, self.step, self.ckpt_step = None, 0, 0
        self.use_onnx = use_onnx
        self.pending_create_workers = 0
        self.ckpt_steps = []
        self.step_cond = asyncio.Condition()
        self.worker_cond = asyncio.Condition()


@ray.remote(num_cpus=0)
class Model:
    def __init__(
        self,
        name: str,
        torch_model: torch.nn.Module = None,
        forward_args: tuple[np.ndarray] = None,
        forward_kwargs: dict[str : np.ndarray] = None,
        checkpoint_interval: int = None,
        max_batch_size: int = 1,
        num_workers: int = None,
        cpus_per_worker: float = 0.5,
        gpus_per_worker: float = 0.0,
        memory_per_worker: int = 1024,
        use_onnx: str = None,
    ):
        self.trial_path = ray.get_runtime_context().namespace
        root_path = os.path.join(self.trial_path, f"{name}")
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)

        torch_path = os.path.join(root_path, f"{name}.pt")
        if not os.path.exists(torch_path):
            assert torch_model is not None, "Missing torch model"
            torch.save(torch_model, torch_path)
        else:
            print("Loading model from", torch_path)
            torch_model = torch.load(torch_path)
        self.name, self.model = name, ray.put(torch_model)

        args_path = os.path.join(root_path, "forward_inputs.pt")
        if not os.path.exists(args_path):
            assert forward_args or forward_kwargs, "Missing forward args"
            torch.save((forward_args, forward_kwargs), args_path)
        else:
            forward_args, forward_kwargs = torch.load(args_path)
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs

        self.checkpoint_interval = checkpoint_interval

        self.models: dict[str:ModelMeta] = {}

        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )

        weights = get_torch_model_weights(torch_model)
        self._initialize_model(
            self.name, weights, max_batch_size, num_workers, use_onnx
        )

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

    def _initialize_model(self, name, weights, max_batch_size, num_workers, use_onnx):
        ckpt_dir = os.path.join(self.trial_path, f"{name}/checkpoint")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        weights_path = os.path.join(self.trial_path, f"{name}/weights.pt")
        if not os.path.exists(weights_path):
            tensor_weights = handle_nested_array(weights, torch.from_numpy)
            torch.save(tensor_weights, weights_path)

        ckpt_steps, step = [], 0
        try:
            ckpt_steps = [
                int(ckpt.split(".")[0].split("-")[1]) for ckpt in os.listdir(ckpt_dir)
            ]
            ckpt_steps.sort()
            step = ckpt_steps[-1] if ckpt_steps else 0
            weights = self._load_checkpoint(name, step=step)
        except Exception as e:
            print(f"Load checkpoint failed: {e}")

        meta = ModelMeta(num_workers, use_onnx)
        meta.step, meta.ckpt_step, meta.ckpt_steps = step, step, ckpt_steps
        meta.max_batch_size = max_batch_size
        if meta.step != 0:
            meta.weights = ray.put(weights)
        self.models[name] = meta
        if meta.step > 0:
            print(f"Model {name} load checkpoint at step {step}")

        if meta.use_onnx:
            onnx_model, forward_outputs = self._get_onnx_model(name, meta.use_onnx)
            print(f"Using onnx model for {name} {meta.use_onnx}.")
        else:
            onnx_model, forward_outputs = None, None

        if name != self.name:
            return
        self.onnx_model, self.forward_outputs = onnx_model, forward_outputs

    def _get_onnx_model(self, name, use_onnx):
        onnx_path_postfix = f"{name}/{self.name}-infer.onnx"
        if use_onnx == "train":
            onnx_path_postfix = f"{self.name}/{self.name}.onnx"
        onnx_path = os.path.join(self.trial_path, onnx_path_postfix)
        outputs_path = os.path.join(
            self.trial_path,
            f"{self.name}/forward_outputs.pt",
        )
        if os.path.exists(onnx_path) and os.path.exists(outputs_path):
            with open(onnx_path, "rb") as f:
                onnx_model = f.read()
            forward_outputs = torch.load(outputs_path)
            return onnx_model, forward_outputs

        from bray.model.onnx import export_onnx

        print("Exporting onnx model to", onnx_path)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        torch_model = ray.get(self.model)
        weights = self._load_checkpoint(name, step=0)
        set_torch_model_weights(torch_model, weights)
        forward_outputs = export_onnx(
            torch_model,
            onnx_path,
            self.forward_args,
            self.forward_kwargs,
            export_params=False if use_onnx == "train" else True,
        )
        torch.save(forward_outputs, outputs_path)
        with open(onnx_path, "rb") as f:
            onnx_model = f.read()
        return onnx_model, forward_outputs

    async def _create_worker(self, name):
        meta: ModelMeta = self.models[name]
        if meta.pending_create_workers > (meta.num_workers or 10):
            return
        worker = self.RemoteModelWorker.remote(name)
        meta.pending_create_workers += 1
        if not await self._is_health(worker):
            meta.pending_create_workers -= 1
            return
        meta.pending_create_workers -= 1
        if meta.num_workers is not None and len(meta.workers) >= meta.num_workers:
            return
        worker.__node_id = await worker.get_node_id.remote()
        meta.workers.append(worker)
        async with meta.worker_cond:
            meta.worker_cond.notify_all()

    async def set_weights(self, name, weights: list[ray.ObjectRef]):
        meta: ModelMeta = self.models[name]
        meta.weights = weights[0]
        meta.step += 1
        merge(
            "step",
            meta.step,
            desc={
                "time_window_cnt": "step per minute",
                "time_window_avg": "smoothed current step",
            },
            model=name,
        )
        async with meta.step_cond:
            meta.step_cond.notify_all()
        if (
            self.checkpoint_interval is not None
            and meta.step % self.checkpoint_interval == 0
        ):
            await self.__save_checkpoint(name)

    async def clone(self, name, step, max_batch_size, num_workers, use_onnx) -> str:
        step, weights = self._get_target_step(name, step)

        if (cloned_name := f"{name}/clone-step-{step}") in self.models:
            return cloned_name

        weights = await (weights if weights else self.get_weights(name, step))
        meta: ModelMeta = self.models[name]
        use_onnx = use_onnx if use_onnx != "" else meta.use_onnx
        max_batch_size = (
            max_batch_size if max_batch_size is not None else meta.max_batch_size
        )
        num_workers = num_workers if num_workers != -1 else meta.num_workers

        await asyncio.get_running_loop().run_in_executor(
            None,
            self._initialize_model,
            cloned_name,
            weights,
            max_batch_size,
            num_workers,
            use_onnx,
        )
        for _ in range(
            len([node for node in ray.nodes() if node["Alive"]])
            if num_workers is None
            else num_workers
        ):
            asyncio.create_task(self._create_worker(cloned_name))
        return cloned_name

    async def subscribe_weights(self, name, current_step):
        meta: ModelMeta = self.models[name]
        async with meta.step_cond:
            await meta.step_cond.wait_for(lambda: meta.step > current_step)
        return meta.weights, meta.step

    async def subscribe_workers(self, name, node_id: str = None):
        meta: ModelMeta = self.models[name]
        async with meta.worker_cond:
            await meta.worker_cond.wait()
        return await self.get_workers(name, node_id)

    def _load_balance(self, name):
        meta: ModelMeta = self.models[name]
        if len(meta.workers) == 0:
            asyncio.create_task(self._create_worker(name))
            return
        forward_time_sum = query("forward", kind="sum", model=name)
        local_forward_time_sum = query(
            "forward",
            kind="sum",
            model=name,
            mode="local",
        )
        forward_time_sum -= local_forward_time_sum
        load_rate = max(0.0, forward_time_sum) / (len(meta.workers) * 60 * 1000)
        merge(
            "load",
            load_rate,
            desc={"time_window_avg": "load rate of model forward"},
            model=name,
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
            await worker.forward.remote(*self.forward_args, **self.forward_kwargs)
            return True
        except ray.exceptions.RayActorError:
            return False
        except Exception as e:
            print(f"Worker is not health: ", e)
            return False

    async def __health_check(self, name):
        meta: ModelMeta = self.models[name]
        worker_num = len(meta.workers)
        active_workers = [
            worker
            for worker in meta.workers[:worker_num]
            if await self._is_health(worker)
        ]
        old_workers = meta.workers
        meta.workers = active_workers
        meta.workers.extend(old_workers[worker_num:])
        merge(
            "worker",
            len(meta.workers),
            desc={"time_window_avg": "smoothed model worker num"},
            model=name,
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
        ckpt_dir = os.path.join(self.trial_path, f"{name}/checkpoint")
        if step == 0:
            weights_path = os.path.join(self.trial_path, f"{name}/weights.pt")
        else:
            weights_path = os.path.join(ckpt_dir, f"step-{step}.pt")
        return handle_nested_array(
            torch.load(weights_path), lambda x: x.numpy(), type_check=False
        )

    async def __save_checkpoint(self, name):
        meta: ModelMeta = self.models[name]
        ckpt_dir = os.path.join(self.trial_path, f"{name}/checkpoint")
        ckpt_path = os.path.join(
            ckpt_dir,
            f"step-{meta.step}.pt",
        )
        asyncio.get_running_loop().run_in_executor(
            None,
            torch.save,
            handle_nested_array(await meta.weights, torch.from_numpy),
            ckpt_path,
        )
        meta.ckpt_step = meta.step
        meta.ckpt_steps.append(meta.ckpt_step)

    async def _save_checkpoint(self):
        await asyncio.sleep(10 * 60)  # save checkpoint every 10 minutes
        for name, meta in list(self.models.items()):
            if meta.ckpt_step >= meta.step:
                continue
            await self.__save_checkpoint(name)
        asyncio.create_task(self._save_checkpoint())

    def _get_target_step(self, name, step) -> tuple[int, object]:
        meta: ModelMeta = self.models[name]
        if step == -1 and meta.weights:
            return meta.step, meta.weights
        ckpt_steps = [0] + meta.ckpt_steps
        index = np.searchsorted(ckpt_steps, step, side="right")
        return ckpt_steps[max(0, index - 1)], None

    async def get_weights(self, name, step=-1) -> NestedArray:
        step, weights = self._get_target_step(name, step)
        if weights:
            return await weights
        return await asyncio.get_running_loop().run_in_executor(
            None, self._load_checkpoint, name, step
        )

    async def get_step(self, name) -> int:
        return self.models[name].step

    async def get_model(self) -> ray.ObjectRef:
        return self.model

    async def get_onnx_model(self, name) -> tuple[bytes, NestedArray]:
        meta: ModelMeta = self.models[name]
        if not meta.use_onnx:
            return None, None
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_onnx_model, name, meta.use_onnx
        )

    async def get_max_batch_size(self, name) -> int:
        meta: ModelMeta = self.models[name]
        return meta.max_batch_size

    async def get_use_onnx(self, name) -> bool:
        return self.models[name].use_onnx

    async def get_cpus_gpus_per_worker(self) -> tuple[float, float]:
        return self.cpus_per_worker, self.gpus_per_worker

    async def get_workers(self, name, node_id: str = None) -> list[ModelWorker]:
        model: ModelMeta = self.models[name]
        if not node_id:
            return model.workers
        workers = [w for w in model.workers if w.__node_id == node_id]
        return workers if workers else model.workers


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个ModelWorker实现并行计算
    """

    remote_models: dict[str:"RemoteModel"] = {}
    max_cached_model: int = 10

    set_executor: bool = False

    def __new__(
        cls,
        name: str = None,
        model: torch.nn.Module = None,
        forward_args: tuple[np.ndarray] = (),
        forward_kwargs: dict[str : np.ndarray] = {},
        checkpoint_interval: int = None,
        max_batch_size: int = 1,
        num_workers: int = None,
        cpus_per_worker: float = 0.5,
        gpus_per_worker: float = 0.0,
        memory_per_worker: int = 1024,
        use_onnx: ["train", "infer"] = None,
        local_mode: bool = False,
    ):
        if name in cls.remote_models and (
            (self := cls.remote_models[name]) is not None
        ):
            return self
        self = super().__new__(cls)
        if name is None:  # 适配对象反序列化时调用__new__方法
            return self
        self.name = name
        base_name = name.split("/")[0]
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.subscribe_task = None
        self.local = local_mode
        self.local_worker = None
        self.model = Model.options(
            name=base_name, get_if_exists=True, scheduling_strategy=scheduling_local
        ).remote(
            base_name,
            model,
            forward_args,
            forward_kwargs,
            checkpoint_interval,
            max_batch_size,
            num_workers,
            cpus_per_worker,
            gpus_per_worker,
            memory_per_worker,
            use_onnx,
        )
        self.workers = ray.get(
            self.model.get_workers.remote(name, ray.get_runtime_context().get_node_id())
        )
        self.worker_index = random.randint(0, len(self.workers))
        cls.remote_models[name] = self
        names = list(cls.remote_models.keys())
        if len(names) > cls.max_cached_model:
            cls.remote_models.pop(names[0])  # pop the oldest one
        return self

    def __init__(self, name: str, *args, **kwargs):
        """
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            forward_args: 模型forward的位置参数输入，用于初始化模型
            forward_kwargs: 模型forward的关键字参数输入，用于初始化模型
            checkpoint_interval: 模型的checkpoint间隔，单位step，默认10分钟保存一次
            max_batch_size: 模型的max_batch_size
            num_workers: 模型的worker数量，如果为None，则会自动根据负载情况调整
            cpus_per_worker: 每个worker的CPU核心数
            gpus_per_worker: 每个worker的GPU数量，如果为0，则表示不使用GPU
            memory_per_worker: 每个worker的内存大小，单位MB
            use_onnx: 默认不适用onnx优化，可选值为["train", "infer"]，分别表示训练和部署模式
            local_mode: 如果为True，则模型会在本地运行，否则会在Ray集群中运行
        """
        assert (
            name in RemoteModel.remote_models
        ), f"RemoteModel {name} is not initialized"

    def __del__(self):
        self.subscribe_task.cancel() if self.subscribe_task else None
        if not self.local_worker:
            return
        self.local_worker.forward_task.cancel()
        self.local_worker.subscribe_task.cancel()

    async def subscribe_workers(cls, model, name, workers):
        # 定义为类方法，避免引用self阻止RemoteModel被回收
        while True:
            workers[:] = await model.subscribe_workers.remote(
                name, ray.get_runtime_context().get_node_id()
            )

    async def _local_forward(self, *args, **kwargs) -> NestedArray:
        loop = asyncio.get_running_loop()

        if not RemoteModel.set_executor:
            loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
            RemoteModel.set_executor = True

        async def build_local_worker():
            self.local_worker = await loop.run_in_executor(
                None, ModelWorker, self.name, loop
            )

        if self.local_worker is None:
            self.local_worker = loop.create_task(build_local_worker())

        local_worker = self.local_worker
        if isinstance(local_worker, asyncio.Task):
            try:
                return await self._remote_forward(*args, **kwargs)
            except:
                pass
            try:
                await local_worker
            except:
                self.local_worker = None
                raise
            assert isinstance(self.local_worker, ModelWorker)

        beg = time.time()
        outputs = await self.local_worker.forward(*args, **kwargs)
        forward_time_ms = (time.time() - beg) * 1000
        merge("forward", forward_time_ms, model=self.name, desc={}, mode="local")
        return outputs

    async def _init_subscribe_task(self):
        if len(self.workers) != 0 and self.subscribe_task:
            return
        if self.subscribe_task:
            await asyncio.sleep(1)
            assert self.workers, f"No model worker for {self.name}"
        self.subscribe_task = asyncio.create_task(
            RemoteModel.subscribe_workers(
                RemoteModel, self.model, self.name, self.workers
            )
        )
        self.subscribe_task.add_done_callback(
            lambda t: None if t.cancelled() else t.result()
        )
        await self.sync()
        await self._init_subscribe_task()

    async def _remote_forward(self, *args, **kwargs) -> NestedArray:
        if len(self.workers) == 0 or not self.subscribe_task:
            await self._init_subscribe_task()
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        if tick_id := get_tick_id():
            index = tick_id % len(self.workers)
        worker = self.workers[index]
        beg = time.time()
        try:
            outputs = await worker.forward.remote(*args, **kwargs)
        except ray.exceptions.RayActorError:
            print("Ray actor exception from model forward")
            await self.sync()
            return await self._remote_forward(*args, **kwargs)
        merge_time_ms("forward", beg, model=self.name, mode="remote")
        return outputs

    async def forward(
        self, *args: tuple[np.ndarray], **kwargs: dict[str : np.ndarray]
    ) -> tuple[np.ndarray] | np.ndarray | dict[str : np.ndarray]:
        """
        执行模型的前向计算，返回模型的输出
        Args:
            *args: 模型的位置参数输入，为一个或者多个 np.ndarray
            **kwargs: 模型的关键字参数输入，为 np.ndarray 字典
        Returns:
            模型的输出，是一个或多个 np.ndarray
        """
        if self.local:
            return await self._local_forward(*args, **kwargs)
        return await self._remote_forward(*args, **kwargs)

    @property
    def step(self) -> int:
        """
        获取模型的最新版本号，每次调用 `remote_model.publish_weights` 会增加版本号
        Returns:
            模型的版本号
        """
        if isinstance(self.local_worker, ModelWorker):
            return self.local_worker.get_model_step()
        if not self.workers:
            return ray.get(self.model.get_step.remote(self.name))
        worker = self.workers[random.randint(0, len(self.workers))]
        return ray.get(worker.get_model_step.remote())

    def get_model(self) -> torch.nn.Module:
        """
        获取被封装的原始模型，权重为最新权重，在Trainer里面会用到
        Returns:
            被封装的Pytorch模型，权重为最新的权重
        """
        torch_model = ray.get(ray.get(self.model.get_model.remote()))
        weights = ray.get(self.model.get_weights.remote(self.name))
        set_torch_model_weights(torch_model, weights)
        return torch_model

    def clone(
        self,
        step: int = -1,
        max_batch_size: int = None,
        num_workers: int = -1,
        use_onnx: ["train", "infer"] = "",
        local_mode: bool = None,
    ) -> "RemoteModel":
        """
        克隆一个新的RemoteModel，可以用于SelfPlay和League的多智能体对抗
        Args:
            step: 克隆的模型的版本号，-1表示克隆最新版本
            local_mode: 是否使用本地模式，None表示使用原来的配置
            max_batch_size: 克隆的模型的max_batch_size，None表示使用原来的
            num_workers: 克隆的模型的worker数量，-1表示使用原来的worker数量， None表示自动负载均衡
            use_onnx: 克隆的模型是否使用onnx，""表示使用原来的配置
        Returns:
            克隆的RemoteModel
        """
        local_mode = local_mode if local_mode is not None else self.local
        cloned_name = ray.get(
            self.model.clone.remote(
                self.name, step, max_batch_size, num_workers, use_onnx
            )
        )
        remote_model = RemoteModel(cloned_name, local_mode=local_mode)
        return remote_model

    def publish_weights(self, weights: NestedArray):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个NestedArray数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        # self.model.set_weights.remote(self.name, [ray.put(weights)])
        self.model.set_weights.remote(
            self.name,
            [ray.put(weights, _owner=self.model)],
        )

    async def sync(self):
        self.workers[:] = await self.model.get_workers.remote(
            self.name, ray.get_runtime_context().get_node_id()
        )
