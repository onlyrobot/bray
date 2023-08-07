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
            print(f"Buffer {self.name} is full")
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
        self.worker_cond = asyncio.Condition()
        asyncio.create_task(self._health_check())

    async def register(self, worker: BufferWorker):
        self.workers.append(worker)
        async with self.worker_cond:
            self.worker_cond.notify_all()

    def get_workers(self) -> tuple[list[BufferWorker], int]:
        return self.workers

    async def _is_health(self, worker):
        try:
            await worker.push.remote()
            return True
        except ray.exceptions.RayActorError:
            return False
        except Exception as e:
            print(f"Worker is not health: ", e)
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

    async def subscribe_workers(self):
        async with self.worker_cond:
            await self.worker_cond.wait(len(self.workers) > 0)
        return self.workers


class RemoteBuffer:
    def __init__(self, name: str, size: int = 1024, no_drop: bool = False):
        self.name, self.size = name, size
        self.no_drop = no_drop
        self.buffer = Buffer.options(name=name, get_if_exists=True).remote()
        self.workers = ray.get(self.buffer.get_workers.remote())
        self.worker_index = 0
        self.buffer_worker = None
        self.subscribe_task = None

    def __del__(self):
        self.subscribe_task.cancel() if self.subscribe_task else None

    async def subscribe_workers(cls, buffer, workers):
        # 定义为类方法，避免引用self阻止RemoteBuffer被回收
        while True:
            workers[:] = await buffer.subscribe_workers.remote()

    async def __push(self, worker, replays):
        async def push():
            try:
                await worker.push.remote(*replays)
            except ray.exceptions.RayActorError:
                await self.sync()
                await self.push(*replays)

        if self.no_drop:
            return await push()
        asyncio.create_task(push())

    async def _push(self, *replays: NestedArray) -> None:
        if not self.subscribe_task:
            await self.sync()
            buffer, workers = self.buffer, self.workers
            self.subscribe_task = asyncio.create_task(
                RemoteBuffer.subscribe_workers(RemoteBuffer, buffer, workers)
            )
        if len(self.workers) == 0:
            await self.sync()
        if len(self.workers) == 0:
            print(f"No available workers for buffer {self.name}")
            return
        step = max(len(replays) // len(self.workers), 1)
        tasks = [
            self.__push(
                self.workers[(self.worker_index + i) % len(self.workers)],
                replays[i : i + step],
            )
            for i in range(0, len(replays), step)
        ]
        await asyncio.gather(*tasks)
        self.worker_index += len(tasks)

    def push(self, *replays: NestedArray) -> None:
        asyncio.create_task(self._push(*replays))

    async def sync(self):
        if not self.no_drop:
            self.workers = await self.buffer.get_workers.remote()
            return
        self.workers = await self.buffer.subscribe_workers.remote()

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
            self.replays, self.last_size, self.next_replays = [], 0, None
        size = len(self.replays)
        if not self.next_replays and size <= self.last_size // 2:
            self.next_replays = self.buffer_worker.pop.remote()
        if size == 0:
            self.replays = ray.get(self.next_replays)
            self.next_replays, self.last_size = None, len(self.replays)
        return self.replays.pop()

    def __iter__(self) -> Iterator[NestedArray]:
        return self
