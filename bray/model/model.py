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
from bray.metric.metric import merge, query, flush_metrics_to_remote


def get_torch_model_weights(model: torch.nn.Module) -> NestedArray:
    return [p.cpu().detach().numpy() for p in model.parameters()]


def set_torch_model_weights(model: torch.nn.Module, weights: NestedArray):
    for p, w in zip(model.parameters(), weights):
        p.copy_(torch.from_numpy(w))


@ray.remote
def save_torch_model_weights(weights: NestedArray, path: str):
    torch.save(weights, path)


@ray.remote
class TorchModelWorker:
    async def __init__(self, name: str):
        self.name, self.model = name, ray.get_actor(name)
        self.forward_time_sum = 0.0
        weights, self.current_step = await self.model.get_weights.remote()
        torch_model = ray.get(self.model.get_model.remote(step=-1))
        self.torch_model = torch_model
        torch_model.requires_grad_(False)
        torch_model.eval()
        set_torch_model_weights(torch_model, weights)
        asyncio.create_task(self._subscribe_weights())
        asyncio.create_task(self._load_balance())

    async def forward(self, inputs: NestedArray) -> NestedArray:
        inputs = make_batch([inputs])
        inputs = handle_nested_array(inputs, torch.from_numpy)
        beg = time.time()
        outputs = self.torch_model(inputs)
        forward_time = (time.time() - beg) * 1000
        self.forward_time_sum += forward_time
        merge(
            "forward",
            forward_time,
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
        set_torch_model_weights(self.torch_model, weights)
        asyncio.create_task(self._subscribe_weights())

    async def _load_balance(self):
        await asyncio.sleep(60)
        worker_num = len(await self.model.get_workers.remote())
        load_rate = await self.model.load_rate.remote()
        local_load_rate = self.forward_time_sum / (60 * 1000)
        # 假设以概率p下掉当前worker，那么下掉后的worker数量为(1-p)*worker_num
        # 目标负载率为0.75，那么下掉后的负载量为(1-p)*worker_num*0.75
        # 它应该等于当前测得的负载量，
        # 即(1-p)*worker_num*0.75 == worker_num * load_rate
        # 解得p = 1 - load_rate / 0.75
        # 为了避免过度下掉，我们加入平滑因子 0.5
        p = 1 - load_rate / 0.75
        if (
            worker_num > 1
            and load_rate * local_load_rate < 0.5
            and random.random() < p * 0.5
        ):
            flush_metrics_to_remote()
            ray.actor.exit_actor()
        self.forward_time_sum = 0.0
        asyncio.create_task(self._load_balance())


@ray.remote
class Model:
    async def __init__(
        self, name: str, torch_model: torch.nn.Module, inputs: NestedArray
    ):
        self.name, self.torch_model = name, torch_model
        assert inputs is not None, "model inputs must be provided"
        self.inputs = inputs
        self.weights = get_torch_model_weights(torch_model)
        self.step = 0
        self.step_cond = asyncio.Condition()
        self.workers = [TorchModelWorker.remote(self.name) for _ in range(4)]
        asyncio.create_task(self._health_check())
        self.ckpt_step = 0
        trial_path = ray.get_runtime_context().namespace
        self.ckpt_dir = os.path.join(trial_path, f"checkpoint/{self.name}")
        if os.path.exists(self.ckpt_dir):
            try:
                self._load_checkpoint()
            except Exception as e:
                print(f"load checkpoint failed: {e}")
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        asyncio.create_task(self._save_checkpoint())

    async def set_weights(self, weights: NestedArray):
        self.weights = weights
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

    def load_rate(self) -> float:
        forward_time_sum = query("forward", kind="sum", model=self.name)
        total_time_sum = len(self.workers) * 60 * 1000
        return forward_time_sum / total_time_sum

    def _load_balance(self):
        # 三级调控，保证负载均衡响应速度，同时避免过度调控
        load_rate = self.load_rate()
        merge(
            "load",
            load_rate,
            desc={"time_window_avg": "load rate of model forward"},
            model=self.name,
        )
        if load_rate < 0.8:
            return
        self.workers.append(TorchModelWorker.remote(self.name))
        if load_rate > 0.9:
            self.workers.append(TorchModelWorker.remote(self.name))
        if load_rate < 0.95:
            return
        self.workers.append(TorchModelWorker.remote(self.name))

    async def _is_health(self, worker):
        try:
            await worker.forward.remote(self.inputs)
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
        self.weights = torch.load(ckpt_path)
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
        self, name: str, model: torch.nn.Module = None, inputs: NestedArray = None
    ):
        """
        Args:
            name: 模型的名字，用于在Ray集群中标识模型
            model: 目前支持PyTorch模型，如果为None，则默认已经存在的同名模型
            inputs: 模型的输入，用于初始化模型和验证模型正确性
        """
        self.name, self.inputs = name, inputs
        self.model = Model.options(
            name=name, get_if_exists=True, lifetime="detached"
        ).remote(name, model, inputs)
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
        if len(self.workers) == 0:
            self.sync()
        if len(self.workers) == 0:
            raise RuntimeError("No available workers")
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

    def publish_weights(self, weights: NestedArray):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个numpy数组
            step: 权重的版本号，每次更新权重都需要增加版本号
        """
        self.model.set_weights.remote(weights)

    def sync(self):
        self.workers = ray.get(self.model.get_workers.remote())
