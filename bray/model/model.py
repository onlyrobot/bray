import asyncio
import time
import random
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import torch
import numpy as np
from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
    flatten_nested_array,
    unflatten_nested_array,
)
from bray.metric.metric import merge, query, merge_time_ms, flush_metrics_to_remote
from bray.actor.actor import get_tick_id


def get_torch_model_weights(model: torch.nn.Module) -> NestedArray:
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.from_numpy(w))


class ModelWorker:
    def __init__(self, name: str, loop: asyncio.AbstractEventLoop = None):
        self.name, self.model = name, ray.get_actor(name)
        self.current_step = ray.get(self.model.get_step.remote())

        self.torch_model, self.ort_session = None, None
        if onnx_model := ray.get(ray.get(self.model.get_onnx_model.remote())):
            self._init_onnx(onnx_model)
        else:
            self._init_torch()

        loop = loop if loop else asyncio.get_running_loop()
        loop.create_task(self._subscribe_weights())

    def _init_onnx(self, onnx_model: bytes):
        import onnxruntime as ort

        ort_session = ort.InferenceSession(onnx_model)
        self.ort_session = ort_session
        self.forward_outputs = ray.get(ray.get(self.model.get_forward_outputs.remote()))

    def _init_torch(self):
        model = ray.get(ray.get(self.model.get_model.remote()))
        model.requires_grad_(False)
        model.eval()
        weights = ray.get(self.model.get_weights.remote())
        set_torch_model_weights(model, ray.get(weights))
        self.torch_model = model

    def _forward_torch(self, *args, **kwargs) -> NestedArray:
        args, kwargs = make_batch([(args, kwargs)])
        args = handle_nested_array(args, torch.from_numpy)
        kwargs = handle_nested_array(kwargs, torch.from_numpy)
        outputs = self.torch_model(*args, **kwargs)
        return handle_nested_array(
            outputs, lambda x: x.squeeze(0).numpy(), type_check=False
        )

    def _forward_onnx(self, *args, **kwargs) -> NestedArray:
        args, kwargs = make_batch([(args, kwargs)])
        sess = self.ort_session
        input_names = [input.name for input in sess.get_inputs()]
        flatten_input = flatten_nested_array(args + (kwargs,), sort_keys=True)
        inputs = dict(zip(input_names, flatten_input))
        # output_names = [output.name for output in sess.get_outputs()]
        # print(handle_nested_array(inputs, lambda x: (x.shape, x.dtype)))
        outputs = sess.run(None, inputs)
        outputs = unflatten_nested_array(self.forward_outputs, outputs)
        return handle_nested_array(outputs, lambda x: x.squeeze(0))

    def forward(self, *args, **kwargs) -> NestedArray:
        beg = time.time()
        if self.torch_model:
            outputs = self._forward_torch(*args, **kwargs)
        else:
            outputs = self._forward_onnx(*args, **kwargs)
        merge_time_ms("forward", beg, model=self.name)
        return outputs

    async def _subscribe_weights(self):
        try:
            weights, step = await self.model.subscribe_weights.remote(
                self.current_step,
            )
        except Exception as e:
            flush_metrics_to_remote()
            print(f"Fail to subscribe weights from {self.name}, worker exit.", e)
            ray.kill(ray.get_runtime_context().current_actor)
            raise e
        assert step > self.current_step
        self.current_step = step
        if self.torch_model:
            set_torch_model_weights(self.torch_model, await weights)
        else:
            pass
        asyncio.create_task(self._subscribe_weights())

    def get_node_id(self):
        return ray.get_runtime_context().get_node_id()


@ray.remote(num_cpus=0)
class Model:
    def __init__(
        self,
        name: str,
        torch_model: torch.nn.Module = None,
        forward_args: tuple[np.ndarray] = None,
        forward_kwargs: dict[str : np.ndarray] = None,
        num_workers: int = None,
        cpus_per_worker: float = 0.5,
        memory_per_worker: int = 1024,
        use_onnx: bool = None,
    ):
        self.trial_path = ray.get_runtime_context().namespace
        names = name.split("/")
        root_path = os.path.join(self.trial_path, f"{names[0]}")
        torch_path = os.path.join(root_path, f"{names[0]}.pt")
        if not os.path.exists(torch_path):
            assert torch_model is not None, "torch model must be provided"
            os.makedirs(os.path.dirname(torch_path), exist_ok=True)
            torch.save(torch_model, torch_path)
        else:
            print("Loading model from", torch_path)
            torch_model = torch.load(torch_path)
        self.name, self.model = name, ray.put(torch_model)

        args_path = os.path.join(root_path, f"forward_inputs.pt")
        if not os.path.exists(args_path):
            assert forward_args or forward_kwargs, "model inputs must be provided"
            os.makedirs(os.path.dirname(args_path), exist_ok=True)
            torch.save((forward_args, forward_kwargs), args_path)
        else:
            forward_args, forward_kwargs = torch.load(args_path)
        self.forward_args, self.forward_kwargs = forward_args, forward_kwargs

        weights, step = get_torch_model_weights(torch_model), 0
        weights_path = os.path.join(self.trial_path, f"{self.name}/weights.pt")
        if os.path.exists(weights_path):
            weights, step = torch.load(weights_path), 0
        self.ckpt_dir = os.path.join(self.trial_path, f"{self.name}/checkpoint")
        if os.path.exists(self.ckpt_dir):
            try:
                weights, step = self._load_checkpoint(step=-1)
            except Exception as e:
                print(f"load checkpoint failed: {e}")
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.weights, self.step, self.ckpt_step = ray.put(weights), step, step
        if self.step > 0:
            print(f"model {self.name} load checkpoint at step {step}")

        onnx_path = os.path.join(root_path, f"{names[0]}.onnx")
        outputs_path = os.path.join(root_path, f"forward_outputs.pt")
        onnx_path_exists = os.path.exists(onnx_path) and os.path.exists(outputs_path)
        if use_onnx and not onnx_path_exists:
            from bray.model.onnx import export_onnx

            print("Exporting onnx model to", onnx_path)
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            forward_outputs = export_onnx(
                torch_model, onnx_path, forward_args, forward_kwargs
            )
            torch.save(forward_outputs, outputs_path)
            onnx_path_exists = True
        if use_onnx is not False and onnx_path_exists:
            print("Loading onnx model from", onnx_path)
            with open(onnx_path, "rb") as f:
                self.onnx_model = ray.put(f.read())
            self.forward_outputs = ray.put(torch.load(outputs_path))
        else:
            self.onnx_model = ray.put(None)

        self.cpus_per_worker = cpus_per_worker
        self.memory_per_worker = memory_per_worker

        self.RemoteModelWorker = ray.remote(ModelWorker).options(
            num_cpus=cpus_per_worker,
            memory=memory_per_worker * 1024 * 1024,
            scheduling_strategy="SPREAD",
        )

        self.num_workers, self.workers = num_workers, []

        if len(names) > 1:  # cloned model
            worker = ray.remote(ModelWorker).remote(self.name)
            worker.__node_id = ray.get(worker.get_node_id.remote())
            self.workers.append(worker)

        for _ in range(
            num_workers
            if num_workers
            else len([node for node in ray.nodes() if node["Alive"]])
        ):
            asyncio.create_task(self._create_worker())

        self.step_cond = asyncio.Condition()
        self.worker_cond = asyncio.Condition()
        asyncio.create_task(self._health_check())
        asyncio.create_task(self._save_checkpoint())

    async def _create_worker(self):
        worker = self.RemoteModelWorker.remote(self.name)
        if not await self._is_health(worker):
            return
        if self.num_workers and len(self.workers) >= self.num_workers:
            return
        worker.__node_id = await worker.get_node_id.remote()
        self.workers.append(worker)
        async with self.worker_cond:
            self.worker_cond.notify_all()

    async def set_weights(self, weights: list[ray.ObjectRef]):
        self.weights = weights[0]
        self.step += 1
        merge(
            "step",
            self.step,
            desc={
                "time_window_cnt": "step per minute",
                "time_window_avg": "smoothed current step",
            },
            model=self.name,
        )
        async with self.step_cond:
            self.step_cond.notify_all()

    async def _keep_clone_model_alive(self, name, model):
        await asyncio.sleep(60)
        forward_cnt = query("forward", kind="cnt", model=name)
        if forward_cnt < 2 and len(await model.get_workers.remote()) < 2:
            print(f"model {name} is not used, drop it")
            return
        asyncio.create_task(self._keep_clone_model_alive(name, model))

    async def clone(self, step) -> str:
        step = self.step if step == -1 else step
        name = self.name + f"/clone-step-{step}"
        try:
            ray.get_actor(name)
            return name
        except ValueError:
            pass

        weights_path = os.path.join(self.trial_path, f"{name}/weights.pt")
        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            weights, ckpt_step = self._load_checkpoint(step)
            torch.save(weights, weights_path)
            print(f"clone model {self.name} to {name} at step {ckpt_step}")
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        model = Model.options(
            name=name,
            get_if_exists=True,
            scheduling_strategy=scheduling_local,
        ).remote(
            name,
            cpus_per_worker=self.cpus_per_worker,
            memory_per_worker=self.memory_per_worker,
        )
        asyncio.create_task(self._keep_clone_model_alive(name, model))
        return name

    async def subscribe_weights(self, current_step):
        async with self.step_cond:
            await self.step_cond.wait_for(lambda: self.step > current_step)
        return self.weights, self.step

    async def subscribe_workers(self, node_id: str = None):
        async with self.worker_cond:
            await self.worker_cond.wait()
        return await self.get_workers(node_id)

    def _load_balance(self):
        if len(self.workers) == 0:
            asyncio.create_task(self._create_worker())
            return
        forward_time_sum = query("forward", kind="sum", model=self.name)
        local_forward_time_sum = query(
            "forward", kind="sum", model=self.name, mode="local"
        )
        forward_time_sum -= local_forward_time_sum
        load_rate = max(0.0, forward_time_sum) / (len(self.workers) * 60 * 1000)
        merge(
            "load",
            load_rate,
            desc={"time_window_avg": "load rate of model forward"},
            model=self.name,
        )
        # 假设以概率p下掉worker，那么下掉后的worker数量为(1-p)*worker_num
        # 目标负载率为0.6，那么下掉后的负载量为(1-p)*worker_num*0.6
        # 它应该等于当前测得的负载量，
        # 即(1-p)*worker_num*0.6 == worker_num * load_rate
        # 解得p = 1 - load_rate / 0.6
        # 为了避免过度下掉，我们加入平滑因子 random.random() ** 3
        if load_rate < 0.4 and len(self.workers) > 1:
            p = 1 - load_rate / 0.6
            shrink_num = int(p * random.random() ** 3 * len(self.workers))
            del self.workers[len(self.workers) - shrink_num :]
            return
        if load_rate < 0.55:
            return
        # 三级调控，保证负载均衡响应速度，同时避免过度调控
        add_rate = 0.5 if load_rate < 0.7 else (1 if load_rate < 0.8 else 1.5)
        for _ in range(1 + int(add_rate * len(self.workers))):
            asyncio.create_task(self._create_worker())

    async def _is_health(self, worker):
        try:
            await worker.forward.remote(*self.forward_args, **self.forward_kwargs)
            return True
        except ray.exceptions.RayActorError:
            return False
        except Exception as e:
            print(f"worker is not health: ", e)
            return False

    async def _health_check(self):
        worker_num = len(self.workers)
        active_workers = [
            worker
            for worker in self.workers[:worker_num]
            if await self._is_health(worker)
        ]
        old_workers = self.workers
        self.workers = active_workers
        self.workers.extend(old_workers[worker_num:])
        merge(
            "worker",
            len(self.workers),
            desc={"time_window_avg": "smoothed model worker num"},
            model=self.name,
        )
        if not self.num_workers:
            self._load_balance()
        elif len(self.workers) < self.num_workers:
            asyncio.create_task(self._create_worker())
        await asyncio.sleep(60)
        asyncio.create_task(self._health_check())

    def _load_checkpoint(self, step=-1) -> tuple[NestedArray, int]:
        ckpts = os.listdir(self.ckpt_dir)
        ckpts = [int(ckpt.split(".")[0].split("-")[1]) for ckpt in ckpts]
        if not ckpts:
            return get_torch_model_weights(ray.get(self.model)), 0
        ckpts.sort()
        if step == -1:
            ckpt_step = ckpts[-1]
        else:
            index = np.searchsorted(ckpts, step, side="right")
            ckpt_step = ckpts[max(0, index - 1)]
        ckpt_path = os.path.join(self.ckpt_dir, f"step-{ckpt_step}.pt")
        return torch.load(ckpt_path), ckpt_step

    async def _save_checkpoint(self):
        await asyncio.sleep(10 * 60)
        asyncio.create_task(self._save_checkpoint())
        if self.ckpt_step >= self.step:
            return
        ckpt_path = os.path.join(
            self.ckpt_dir,
            f"step-{self.step}.pt",
        )
        await asyncio.get_running_loop().run_in_executor(
            None, torch.save, await self.weights, ckpt_path
        )
        self.ckpt_step = self.step

    def __del__(self):
        self.workers.clear()
        flush_metrics_to_remote()

    async def get_weights(self) -> ray.ObjectRef:
        return self.weights

    async def get_step(self) -> int:
        return self.step

    async def get_model(self) -> ray.ObjectRef:
        return self.model

    async def get_onnx_model(self) -> ray.ObjectRef:
        return self.onnx_model

    async def get_forward_outputs(self) -> ray.ObjectRef:
        return self.forward_outputs

    async def get_workers(self, node_id: str = None) -> list[ModelWorker]:
        if not node_id:
            return self.workers
        workers = [w for w in self.workers if w.__node_id == node_id]
        return workers if workers else self.workers


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个ModelWorker实现并行计算
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module = None,
        forward_args: tuple[np.ndarray] = (),
        forward_kwargs: dict[str : np.ndarray] = {},
        local_mode: bool = False,
        num_workers: int = None,
        cpus_per_worker: float = 0.5,
        memory_per_worker: int = 1024,
        use_onnx: bool = None,
    ):
        """
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            forward_args: 模型forward的位置参数输入，用于初始化模型
            forward_kwargs: 模型forward的关键字参数输入，用于初始化模型
            local_mode: 如果为True，则模型会在本地运行，否则会在Ray集群中运行
            num_workers: 模型的worker数量，如果为None，则会自动根据负载情况调整
            cpus_per_worker: 每个worker的CPU核心数
            memory_per_worker: 每个worker的内存大小，单位MB
            use_onnx: 如果为True，则model会被导出onnx格式，使用onnxruntime推理
        """
        self.name = name
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.model = Model.options(
            name=name, get_if_exists=True, scheduling_strategy=scheduling_local
        ).remote(
            name,
            model,
            forward_args,
            forward_kwargs,
            num_workers,
            cpus_per_worker,
            memory_per_worker,
            use_onnx,
        )
        self.workers = ray.get(
            self.model.get_workers.remote(ray.get_runtime_context().get_node_id())
        )
        self.worker_index = random.randint(0, len(self.workers))
        self.local = local_mode
        self.local_worker = None
        self.already_subscribe = False

    async def _local_forward(self, *args, **kwargs) -> NestedArray:
        loop = asyncio.get_running_loop()

        async def build_local_worker():
            self.local_worker = await loop.run_in_executor(
                None, ModelWorker, self.name, loop
            )
            self.local = True

        if self.local_worker is None:
            self.local = False
            outputs = await self._remote_forward(*args, **kwargs)
            loop.create_task(build_local_worker())
            return outputs

        def forward():
            return self.local_worker.forward(*args, **kwargs)

        beg = time.time()
        outputs = await loop.run_in_executor(None, forward)
        forward_time_ms = (time.time() - beg) * 1000
        merge("forward", forward_time_ms, model=self.name, desc={}, mode="local")
        return outputs

    async def _remote_forward(self, *args, **kwargs) -> NestedArray:
        if not self.already_subscribe:
            self.already_subscribe = True
            asyncio.create_task(self._subscribe_workers())
        if len(self.workers) == 0:
            await self.sync()
        if len(self.workers) == 0:
            raise RuntimeError("No available workers")
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        if tick_id := get_tick_id():
            index = tick_id % len(self.workers)
        try:
            worker = self.workers[index]
            beg = time.time()
            outputs = await worker.forward.remote(*args, **kwargs)
            merge(
                "forward",
                (time.time() - beg) * 1000,
                desc={},
                model=self.name,
                mode="remote",
            )
        except ray.exceptions.RayActorError:
            print("ray actor exception from model forward")
            await self.sync()
            outputs = await self._remote_forward(*args, **kwargs)
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
        return ray.get(self.model.get_step.remote())

    def get_model(self) -> torch.nn.Module:
        """
        获取被封装的原始模型，权重为最新权重，在Trainer里面会用到
        Returns:
            被封装的Pytorch模型，权重为最新的权重
        """
        torch_model = ray.get(ray.get(self.model.get_model.remote()))
        weights = ray.get(ray.get(self.model.get_weights.remote()))
        set_torch_model_weights(torch_model, weights)
        return torch_model

    def clone(self, step: int = -1) -> "RemoteModel":
        """
        克隆一个新的RemoteModel，用于SelfPlay和League的多智能体对抗
        Args:
            step: 克隆的模型的版本号，-1表示克隆最新版本
        Returns:
            克隆的RemoteModel
        """
        remote_model = RemoteModel(ray.get(self.model.clone.remote(step)))
        remote_model.local = False
        return remote_model

    def publish_weights(self, weights: NestedArray):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个numpy数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        # self.model.set_weights.remote([ray.put(weights)])
        self.model.set_weights.remote([ray.put(weights, _owner=self.model)])

    async def sync(self):
        self.workers = await self.model.get_workers.remote(
            ray.get_runtime_context().get_node_id()
        )

    async def _subscribe_workers(self):
        self.workers = await self.model.subscribe_workers.remote(
            ray.get_runtime_context().get_node_id()
        )
        asyncio.create_task(self._subscribe_workers())
