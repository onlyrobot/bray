import asyncio
import time

import ray
import torch
from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)
from bray.metric.metric import merge


def get_torch_model_weights(model: torch.nn.Module) -> NestedArray:
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    for p, w in zip(model.parameters(), weights):
        p.copy_(torch.from_numpy(w))


@ray.remote
class TorchModelWorker:
    async def __init__(self, name: str):
        self.name, self.model = name, ray.get_actor(name)
        weights, self.current_step = await self.model.get_weights.remote()
        torch_model = ray.get(self.model.get_model.remote(step=-1))
        self.torch_model = torch_model
        torch_model.requires_grad_(False)
        torch_model.eval()
        set_torch_model_weights(torch_model, weights)
        asyncio.create_task(self._subscribe_weights())

    async def forward(self, inputs: NestedArray) -> NestedArray:
        inputs = make_batch([inputs])
        inputs = handle_nested_array(inputs, torch.from_numpy)
        beg = time.time()
        outputs = self.torch_model(inputs)
        merge("forward_time_ms", (time.time() - beg) * 1000, model=self.name)
        return handle_nested_array(
            outputs, lambda x: x.squeeze(0).numpy(), type_check=False
        )

    async def _subscribe_weights(self):
        weights, step = await self.model.subscribe_weights.remote(self.current_step)
        assert step > self.current_step
        self.current_step = step
        set_torch_model_weights(self.torch_model, weights)
        asyncio.create_task(self._subscribe_weights())


@ray.remote
class Model:
    async def __init__(self, name: str, torch_model: torch.nn.Module):
        self.name, self.torch_model = name, torch_model
        self.weights = get_torch_model_weights(torch_model)
        self.step = 0
        self.step_cond = asyncio.Condition()
        self.workers = [TorchModelWorker.remote(self.name) for _ in range(4)]
        asyncio.create_task(self._health_check())

    async def set_weights(self, weights: NestedArray, step):
        step = self.step + 1 if step == -1 else step
        if step <= self.step:
            print(f"Skip set_weights with step={step}, current_step={self.step}")
            return
        self.weights = weights
        self.step = step
        async with self.step_cond:
            self.step_cond.notify_all()

    def get_model(self, step) -> torch.nn.Module:
        if step != -1:
            raise NotImplementedError
        with torch.no_grad():
            set_torch_model_weights(self.torch_model, self.weights)
        return self.torch_model

    def get_workers(self) -> list[TorchModelWorker]:
        return self.workers

    def get_weights(self) -> tuple[NestedArray, int]:
        return self.weights, self.step

    async def subscribe_weights(self, current_step):
        async with self.step_cond:
            await self.step_cond.wait_for(lambda: self.step > current_step)
        return self.weights, self.step

    async def _is_health(self, worker):
        try:
            await worker.forward.remote("fake data here")
            return True
        except ray.exceptions.RayActorError:
            return False
        finally:
            return True

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
        await asyncio.sleep(60)
        asyncio.create_task(self._health_check())


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个TorchModelWorker实现并行计算
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module = None,
        inputs: NestedArray = None,
        override: bool = None,
    ):
        """
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            override: 如果为True，会覆盖已经存在的同名模型
        """
        self.name = name
        self.model = Model.options(
            name=name, get_if_exists=True, lifetime="detached"
        ).remote(name, model)
        self.workers, self.worker_index = [], 0
        self.sync()

    def forward(self, inputs: NestedArray) -> ray.ObjectRef:
        """
        执行模型的前向计算，返回模型的输出
        Args:
            inputs: 模型的输入
        Returns:
            模型的输出，是一个Ray ObjectRef，可以通过ray.get()获取
        """
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        try:
            return self.workers[index].forward.remote(inputs)
        except ray.exceptions.RayActorError:
            self.sync()
            return self.forward(inputs)

    def get_model(self, step: int = -1) -> torch.nn.Module:
        """
        获取被封装的原始模型，在Trainer里面会用到
        Args:
            step: 模型的版本号，如果为-1，会返回最新的模型
        Returns:
            被封装的Pytorch模型
        """
        return ray.get(self.model.get_model.remote(step))

    def get_weights(self) -> tuple[NestedArray, int]:
        """
        获取模型的最新权重和版本号
        Returns:
            模型的权重和版本号
        """
        return ray.get(self.model.get_weights.remote())

    def publish_weights(self, weights: NestedArray, step: int = -1):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个numpy数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        self.model.set_weights.remote(weights, step)

    def sync(self):
        self.workers = ray.get(self.model.get_workers.remote())
