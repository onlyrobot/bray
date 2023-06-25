from typing import Iterator

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import asyncio

from bray.utils.nested_array import NestedArray, make_batch


@ray.remote
class BufferWorker:
    def __init__(self, buffer: "Buffer"):
        self.replays, self.size = [], 256
        self_handle = ray.get_runtime_context().current_actor
        buffer.register.remote(self_handle)

    async def push(self, *replays: NestedArray):
        if len(self.replays) > self.size:
            print("buffer is full")
            return
        self.replays.extend(replays)

    async def pop(self) -> NestedArray:
        while len(self.replays) == 0:
            await asyncio.sleep(0.01)
        return self.replays.pop()


class BufferIterator:
    def __init__(self, buffer_worker: BufferWorker):
        self.buffer_worker = buffer_worker

    def __next__(self) -> NestedArray:
        return ray.get(self.buffer_worker.pop.remote())

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class BatchBuffer:
    def __init__(self, buffer: Iterator[NestedArray], batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __next__(self) -> NestedArray:
        batch = []
        for _ in range(self.batch_size):
            batch.append(next(self.buffer))
        return make_batch(batch)

    def __iter__(self) -> Iterator[NestedArray]:
        return self


class ReuseBuffer:
    pass


class PrefetchBuffer:
    pass


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
    def __init__(self, name: str):
        self.buffer = Buffer.options(name=name, get_if_exists=True).remote()
        self.workers, self.worker_index = [], 0
        self.sync()

    def push(self, *reploys: NestedArray):
        if len(self.workers) == 0:
            self.sync()
        if len(self.workers) == 0:
            print("No buffer worker found, push failed.")
            return
        index = self.worker_index % len(self.workers)
        self.worker_index += 1
        try:
            self.workers[index].push.remote(*reploys)
        except ray.exceptions.RayActorError:
            self.sync()
            self.push(*reploys)

    def sync(self):
        self.workers = ray.get(self.buffer.get_workers.remote())

    def new_local_worker(self) -> BufferWorker:
        """
        创建一个本地的 BufferWorker，这个 BufferWorker 会在当前节点上运行，
        用于从本地的 Buffer 中读取数据，一般被 TrainerWorker 调用
        """
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        return BufferWorker.options(scheduling_strategy=scheduling_local).remote(
            self.buffer
        )
