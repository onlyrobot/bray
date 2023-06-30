import asyncio
import time
import random
import os

import ray
import torch
from bray.utils.nested_array import (
    NestedArray,
    make_batch,
    handle_nested_array,
)
from bray.metric.metric import merge, query
from bray.model.utils import export_onnx


def get_torch_model_weights(model: torch.nn.Module) -> NestedArray:
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.from_numpy(w))


@ray.remote
def save_torch_model_weights(weights: NestedArray, path: str):
    torch.save(weights, path)


class TorchModelWorker:
    def __init__(self, name: str):
        self.name, self.model = name, ray.get_actor(name)
        weights, self.current_step = ray.get(self.model.get_weights.remote())
        torch_model = ray.get(self.model.get_model.remote(step=-1))
        self.torch_model = torch_model
        torch_model.requires_grad_(False)
        torch_model.eval()
        set_torch_model_weights(torch_model, ray.get(weights))
        asyncio.create_task(self._subscribe_weights())

    def forward(self, *args, **kwargs) -> NestedArray:
        args, kwargs = make_batch([args]), make_batch([kwargs])
        args = handle_nested_array(args, torch.from_numpy)
        kwargs = handle_nested_array(kwargs, torch.from_numpy)
        beg = time.time()
        outputs = self.torch_model(*args, **kwargs)
        merge(
            "forward",
            (time.time() - beg) * 1000,
            desc={
                "time_window_avg": "forward latency ms",
                "time_window_cnt": "forward per minute",
            },
            model=self.name,
        )
        return handle_nested_array(
            outputs, lambda x: x.squeeze(0).numpy(), type_check=False
        )

    async def _subscribe_weights(self):
        weights, step = await self.model.subscribe_weights.remote(self.current_step)
        assert step > self.current_step
        self.current_step = step
        set_torch_model_weights(self.torch_model, await weights)
        asyncio.create_task(self._subscribe_weights())


@ray.remote
class Model:
    async def __init__(self, name: str, torch_model: torch.nn.Module, forward_args):
        self.name, self.torch_model = name, torch_model
        assert forward_args, "model inputs must be provided"
        self.forward_args = forward_args
        self.weights = ray.put(get_torch_model_weights(torch_model))
        self.step = 0
        self.step_cond = asyncio.Condition()
        self.workers = [
            ray.remote(TorchModelWorker).remote(self.name) for _ in range(2)
        ]
        asyncio.create_task(self._health_check())
        trial_path = ray.get_runtime_context().namespace
        torch_path = os.path.join(trial_path, f"torch/{self.name}.pt")
        if not os.path.exists(torch_path):
            os.makedirs(os.path.dirname(torch_path), exist_ok=True)
            torch.save(self.torch_model, torch_path)
        self.ckpt_step = 0
        self.ckpt_dir = os.path.join(trial_path, f"checkpoint/{self.name}")
        if os.path.exists(self.ckpt_dir):
            try:
                self._load_checkpoint()
            except Exception as e:
                print(f"load checkpoint failed: {e}")
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        asyncio.create_task(self._save_checkpoint())
        onnx_path = os.path.join(trial_path, f"onnx/{self.name}.onnx")
        if not os.path.exists(onnx_path):
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            # export_onnx(self.torch_model, self.forward_args, onnx_path)

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

    async def get_model(self, step) -> torch.nn.Module:
        if step != -1:
            raise NotImplementedError
        with torch.no_grad():
            set_torch_model_weights(self.torch_model, await self.weights)
        return self.torch_model

    def get_workers(self) -> list[TorchModelWorker]:
        return self.workers

    def get_weights(self) -> tuple[ray.ObjectRef, int]:
        return self.weights, self.step

    async def subscribe_weights(self, current_step):
        async with self.step_cond:
            await self.step_cond.wait_for(lambda: self.step > current_step)
        return self.weights, self.step

    def _load_balance(self):
        if len(self.workers) == 0:
            self.workers.append(ray.remote(TorchModelWorker).remote(self.name))
            return
        forward_time_sum = query("forward", kind="sum", model=self.name)
        load_rate = forward_time_sum / (len(self.workers) * 60 * 1000)
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
        # 为了避免过度下掉，我们加入平滑因子 random.random() * random.random()
        if load_rate < 0.4 and len(self.workers) > 1:
            p = 1 - load_rate / 0.6
            shrink_num = int(p * random.random() * random.random() * len(self.workers))
            del self.workers[len(self.workers) - shrink_num :]
            return
        if load_rate < 0.55:
            return
        # 三级调控，保证负载均衡响应速度，同时避免过度调控
        add_rate = 0.5 if load_rate < 0.7 else (1 if load_rate < 0.8 else 1.5)
        self.workers.extend(
            [
                ray.remote(TorchModelWorker).remote(self.name)
                for _ in range(1 + int(add_rate * len(self.workers)))
            ]
        )

    async def _is_health(self, worker):
        try:
            await worker.forward.remote(*self.forward_args)
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
        self._load_balance()
        merge(
            "worker",
            len(self.workers),
            desc={"time_window_avg": "smoothed model worker num"},
            model=self.name,
        )
        await asyncio.sleep(60)
        asyncio.create_task(self._health_check())

    def _load_checkpoint(self):
        ckpts = os.listdir(self.ckpt_dir)
        ckpts = [int(ckpt.split(".")[0].split("-")[1]) for ckpt in ckpts]
        if not ckpts:
            return
        ckpts.sort()
        self.ckpt_step = ckpts[-1]
        self.step = self.ckpt_step
        ckpt_path = os.path.join(self.ckpt_dir, f"step-{self.ckpt_step}.pt")
        self.weights = ray.put(torch.load(ckpt_path))
        print(f"load checkpoint {ckpt_path}")

    async def _save_checkpoint(self):
        await asyncio.sleep(10 * 60)
        if self.ckpt_step < self.step:
            save_torch_model_weights.remote(
                self.weights,
                os.path.join(
                    self.ckpt_dir,
                    f"step-{self.step}.pt",
                ),
            )
            self.ckpt_step = self.step
        asyncio.create_task(self._save_checkpoint())


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个TorchModelWorker实现并行计算
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module = None,
        forward_args: tuple[NestedArray] = None,
    ):
        """
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            forward_args: 模型forward的参数输入，类型为 tuple[NestedArray]，用于初始化模型
        """
        self.name, self.forward_args = name, forward_args
        self.model = Model.options(
            name=name, get_if_exists=True, lifetime="detached"
        ).remote(name, model, forward_args)
        self.worker_index = 0
        self.workers = ray.get(self.model.get_workers.remote())
        self.local = True
        self.local_worker = None

    async def _local_forward(self, *args, **kwargs) -> NestedArray:
        if self.local_worker is None:
            self.local_worker = TorchModelWorker(self.name)
        return self.local_worker.forward(*args, **kwargs)

    async def forward(self, *args, **kwargs) -> NestedArray:
        """
        执行模型的前向计算，返回模型的输出
        Args:
            *args: 模型的位置参数输入，是一个 NestedArray
            **kwargs: 模型的关键字参数输入，是一个 NestedArray
        Returns:
            模型的输出，是一个 NestedArray
        """
        if self.local:
            return await self._local_forward(*args, **kwargs)
        if len(self.workers) == 0:
            await self.sync()
        if len(self.workers) == 0:
            raise RuntimeError("No available workers")
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        try:
            return await self.workers[index].forward.remote(*args, **kwargs)
        except ray.exceptions.RayActorError:
            await self.sync()
            return await self.forward(*args, **kwargs)

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

    def publish_weights(self, weights: NestedArray):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个numpy数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        self.model.set_weights.remote([ray.put(weights)])

    async def sync(self):
        self.workers = await self.model.get_workers.remote()
