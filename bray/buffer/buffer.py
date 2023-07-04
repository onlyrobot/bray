from typing import Iterator
import random

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import asyncio

from bray.utils.nested_array import NestedArray
from bray.metric.metric import merge


@ray.remote
class BufferWorker:
    def __init__(self, name: str, size: int):
        self.name, self.buffer = name, ray.get_actor(name)
        self.replays, self.size = [], size
        self_handle = ray.get_runtime_context().current_actor
        ray.get(self.buffer.register.remote(self_handle))
        self.pop_cond = asyncio.Condition()

    async def push(self, *replays: NestedArray):
        merge(
            "push",
            len(replays),
            desc={"time_window_sum": "push per minute"},
            buffer=self.name,
        )
        self.replays.extend(replays)
        if len(self.replays) > self.size:
            del self.replays[: -self.size]
            print(f"buffer {self.name} is full")
        async with self.pop_cond:
            self.pop_cond.notify_all()

    async def pop(self) -> list[NestedArray]:
        async with self.pop_cond:
            await self.pop_cond.wait_for(lambda: len(self.replays) > 0)
        replays = self.replays.copy()
        merge(
            "pop",
            len(replays),
            desc={"time_window_sum": "pop per minute"},
            buffer=self.name,
        )
        self.replays.clear()  # 重复使用list，避免频繁的内存分配
        return replays


@ray.remote
class Buffer:
    async def __init__(self):
        self.workers = []
        asyncio.create_task(self._health_check())

    def register(self, worker: BufferWorker):
        self.workers.append(worker)

    def get_workers(self) -> list[BufferWorker]:
        return self.workers

    async def _is_health(self, worker):
        try:
            await worker.push.remote()
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
        await asyncio.sleep(60)
        asyncio.create_task(self._health_check())


class RemoteBuffer:
    def __init__(self, name: str, size: int = 256):
        self.name, self.size = name, size
        self.buffer = Buffer.options(
            name=name, get_if_exists=True, lifetime="detached"
        ).remote()
        self.workers, self.worker_index = [], random.randint(0, 100)
        self.sync()
        self.buffer_worker = None

    def push(self, *replays: NestedArray):
        if len(replays) == 0:
            return
        if len(self.workers) == 0:
            self.sync()
        if len(self.workers) == 0:
            print(f"No buffer worker of {self.name} found, push failed.")
            return
        max_push_size = 100
        if len(replays) > max_push_size:
            self.push(*replays[max_push_size:])
            replays = replays[:max_push_size]
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        try:
            self.workers[index].push.remote(*replays)
        except ray.exceptions.RayActorError:
            self.sync()
            self.push(*replays)

    def sync(self):
        self.workers = ray.get(self.buffer.get_workers.remote())

    def _new_local_worker(self) -> BufferWorker:
        """
        创建一个本地的 BufferWorker，这个 BufferWorker 会在当前节点上运行，
        用于从本地的 Buffer 中读取数据，一般被 TrainerWorker 调用
        """
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        return BufferWorker.options(scheduling_strategy=scheduling_local).remote(
            self.name, self.size
        )

    def __next__(self) -> NestedArray:
        if not self.buffer_worker:
            self.buffer_worker = self._new_local_worker()
            self.replays, self.last_size = [], 0
            self.next_replays = None
        size = len(self.replays)
        if not self.next_replays and size <= self.last_size // 2:
            self.next_replays = self.buffer_worker.pop.remote()
        if size == 0:
            self.replays = ray.get(self.next_replays)
            self.next_replays = None
            self.last_size = len(self.replays)
        return self.replays.pop()

    def __iter__(self) -> Iterator[NestedArray]:
        return self
