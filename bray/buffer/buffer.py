from typing import Iterator
import random
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import asyncio

from bray.utils.nested_array import NestedArray, make_batch
from bray.metric.metric import merge, merge_time_ms


@ray.remote
class BufferWorker:
    def __init__(self, name: str, size: int):
        self.name, self.buffer = name, ray.get_actor(name)
        self.replays, self.size = [], size
        self_handle = ray.get_runtime_context().current_actor
        ray.get(self.buffer.register.remote(self_handle))
        self.pop_cond = asyncio.Condition()

    async def push(self, drop=True, *replays: NestedArray):
        if not replays:
            return
        while not drop and len(self.replays) > self.size:
            await asyncio.sleep(0.01)
        before_size = len(self.replays)
        self.replays.extend(replays)
        merge(
            "push",
            len(replays),
            desc={"time_window_sum": "push per minute"},
            buffer=self.name,
        )
        if drop and len(self.replays) > self.size:
            # print(f"Buffer {self.name} is full")
            del self.replays[: -self.size]
        if before_size != 0:
            return
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
        self.worker_cond = asyncio.Condition()
        self.workers = []
        asyncio.create_task(self._health_check())

    async def register(self, worker: BufferWorker):
        self.workers.append(worker)
        await asyncio.sleep(1)
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
            await self.worker_cond.wait()
        return self.workers


class RemoteBuffer:
    def __init__(self, name: str, size: int = 128):
        self.name, self.size = name, size
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.buffer = Buffer.options(
            name=name, get_if_exists=True, scheduling_strategy=scheduling_local
        ).remote()
        self.workers = ray.get(self.buffer.get_workers.remote())
        self.worker_index = random.randint(0, 100)
        self.buffer_workers = None
        self.subscribe_task = None

    def __del__(self):
        self.subscribe_task.cancel() if self.subscribe_task else None

    async def subscribe_workers(cls, buffer, workers):
        # 定义为类方法，避免引用self阻止RemoteBuffer被回收
        while True:
            workers[:] = await buffer.subscribe_workers.remote()

    async def _init_subscribe_task(self, drop):
        if len(self.workers) != 0 and self.subscribe_task:
            return
        while not drop and len(self.workers) == 0:
            await self.sync()
            await asyncio.sleep(0.01)

        if self.subscribe_task:
            await asyncio.sleep(10)
            assert self.workers, f"No buffer worker for {self.name}"
        self.subscribe_task = asyncio.create_task(
            RemoteBuffer.subscribe_workers(RemoteBuffer, self.buffer, self.workers)
        )
        self.subscribe_task.add_done_callback(
            lambda t: None if t.cancelled() else t.result()
        )
        await self.sync()
        await self._init_subscribe_task(drop)

    async def _push(self, drop: bool, *replays: NestedArray) -> None:
        if len(self.workers) == 0 or not self.subscribe_task:
            await self._init_subscribe_task(drop)
        workers_num = len(self.workers)
        step = max(len(replays) // workers_num, 16)
        tasks = [
            self.workers[(self.worker_index + i) % workers_num].push.remote(
                drop, *replays[i : i + step]
            )
            for i in range(0, len(replays), step)
        ]
        beg = time.time()
        try:
            await asyncio.gather(*tasks)
        except ray.exceptions.RayActorError:
            print("Buffer worker is not health, try to sync buffer")
            await self.sync()
        if not drop:
            merge_time_ms("push_", beg, buffer=self.name)
        self.worker_index += len(tasks)

    def push(self, *replays: NestedArray) -> asyncio.Task:
        return asyncio.create_task(self._push(True, *replays))

    async def sync(self):
        self.workers[:] = await self.buffer.get_workers.remote()

    def _new_local_worker(self) -> BufferWorker:
        """
        创建一个本地的 BufferWorker，这个 BufferWorker 会在当前节点上运行，
        用于从本地的 Buffer 中读取数据，一般被 TrainerWorker 调用
        """
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        return BufferWorker.options(
            scheduling_strategy=scheduling_local, max_concurrency=100000
        ).remote(self.name, self.size)

    def __next__(self) -> NestedArray:
        if not self.buffer_workers:
            self.buffer_workers = [self._new_local_worker() for _ in range(2)]
            self.replays = []
            self.pop_index = -1
            self.last_size, self.next_replays = 0, []
        self.pop_index += 1
        size = len(self.replays) - self.pop_index

        # prefetch from buffer when workers num is 1
        if not self.next_replays and size <= self.last_size // 2:
            for w in self.buffer_workers:
                ref = w.__bray_ref = w.pop.remote()
                self.next_replays.append(ref)

            self.last_size, self.ready_replays = 0, []

        if size == 0:
            for w in self.buffer_workers:
                if w.__bray_ref not in self.ready_replays:
                    continue
                ref = w.__bray_ref = w.pop.remote()
                self.next_replays.append(ref)

            self.ready_replays, self.next_replays = ray.wait(
                self.next_replays, num_returns=1
            )

            self.replays.clear()
            for replays in self.ready_replays:
                self.replays.extend(ray.get(replays))
            self.pop_index = 0
            self.last_size += len(self.replays)

        return self.replays[self.pop_index]

    def __iter__(self) -> Iterator[NestedArray]:
        return self

    async def _generate(self, source: Iterator[NestedArray], batch_size):
        self.worker_index = random.randint(0, 100)
        batch_data, max_batch_size = [], batch_size if batch_size else 1
        last_push_task = asyncio.create_task(asyncio.sleep(0))
        beg = time.time()
        for data in source:
            merge_time_ms("generate", beg, buffer=self.name)
            batch_data.append(data)
            if len(batch_data) < max_batch_size:
                await asyncio.sleep(0)
                beg = time.time()
                continue
            push_task = self._push(
                False,
                *batch_data if batch_size is None else [make_batch(batch_data)],
            )
            push_task = asyncio.create_task(push_task)
            batch_data = []
            await last_push_task
            last_push_task, beg = push_task, time.time()
        await last_push_task
        if not batch_data:
            return
        await self._push(
            False, *batch_data if batch_size is None else [make_batch(batch_data)]
        )

    def add_source(
        self, *sources: Iterator[NestedArray], batch_size=None
    ) -> ray.ObjectRef:
        """
        将一个或多个数据源添加到 Buffer 中，这些数据源会被异步的推送到 Buffer 中

        Args:
            sources: 数据源
            batch_size: 如果为None，则不对数据进行分批，否则会对数据进行分批
        """
        generate = lambda source: asyncio.run(self._generate(source, batch_size))
        generate = ray.remote(generate).options(
            num_cpus=0, scheduling_strategy="SPREAD"
        )
        return [generate.remote(source) for source in sources]
