from typing import Iterator
import random
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import asyncio

from bray.utils.nested_array import NestedArray, make_batch, split_batch
from bray.metric.metric import merge, merge_time_ms


@ray.remote
class BufferWorker:
    def __init__(self, name: str):
        self.name, self.buffer = name, ray.get_actor(name)
        self_handle = ray.get_runtime_context().current_actor
        worker_info = ray.get(self.buffer.register.remote(self_handle))
        self.size, self.batch_size = worker_info
        if self.batch_size is not None:
            self.size = self.size // self.batch_size
        self.replays, self._replays = [], []
        self.pop_cond = asyncio.Condition()

    async def push(self, *replays: NestedArray, drop=True, batch=None):
        if not replays:
            return
        wait_time, wait_interval = 0, 0.01
        while not drop and len(self.replays) >= self.size:
            if wait_time > 1:
                return False
            wait_time += wait_interval
            await asyncio.sleep(wait_interval)
        before_size = len(self.replays)
        merge(
            "push",
            len(replays) * (batch or 1),
            desc={"time_window_sum": "push per minute"},
            buffer=self.name,
        )
        batch_size, _replays = self.batch_size, self._replays
        if batch_size is None or (batch and batch == batch_size):
            self.replays.extend(replays)
        else:
            replays = split_batch(replays[0]) if batch else replays
            _replays.extend(replays)

        while batch_size and len(_replays) >= batch_size:
            batch_data = make_batch(_replays[:batch_size])
            del _replays[:batch_size]
            self.replays.append(batch_data)

        if drop and len(self.replays) > self.size:
            # print(f"Buffer {self.name} is full")
            del self.replays[: -self.size]
        if before_size != 0 or len(self.replays) == 0:
            return
        async with self.pop_cond:
            self.pop_cond.notify_all()

    async def pop(self) -> list[NestedArray]:
        async with self.pop_cond:
            await self.pop_cond.wait_for(lambda: len(self.replays) > 0)
        replays = self.replays.copy()
        batch_size = 1 if self.batch_size is None else self.batch_size
        merge(
            "pop",
            len(replays) * batch_size,
            desc={"time_window_sum": "pop per minute"},
            buffer=self.name,
        )
        self.replays.clear()  # 重复使用list，避免频繁的内存分配
        return replays


@ray.remote
class Buffer:
    def __init__(self, size, batch_size):
        self.size, self.batch_size = size, batch_size
        self.worker_cond = asyncio.Condition()
        self.workers = []
        asyncio.create_task(self._health_check())

    async def register(self, worker: BufferWorker):
        self.workers.append(worker)
        # await asyncio.sleep(1)
        async with self.worker_cond:
            self.worker_cond.notify_all()
        return self.size, self.batch_size

    def get_workers(self, max_num: int) -> list:
        num = min(max_num, len(self.workers))
        return random.sample(self.workers, num)

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

    async def subscribe_workers(self, max_num: int) -> list:
        async with self.worker_cond:
            await self.worker_cond.wait()
        num = min(max_num, len(self.workers))
        return random.sample(self.workers, num)


class RemoteBuffer:
    def __init__(
        self, name: str, size=512, batch_size=None, num_workers=2, sparsity=100
    ):
        """
        创建一个 RemoteBuffer，RemoteBuffer 会在 Ray 集群中运行，用于存储数据
        Args:
            name: Buffer 的名称，用于标识不同的 Buffer
            size: Buffer 的大小，用于限制 Buffer 中最多能存储多少条数据
            batch_size: 从 Buffer 中读取数据时，每次读取的数据的批次大小，
                如果为 None，则不对数据进行分批
            num_workers: BufferWorker 的数量，用于控制从 Buffer 中读取数据的并发度
            sparsity: 开启后Buffer的生产端和消费端不再是全连接，从而减少系统资源占用
        """
        assert (
            batch_size is None or size >= batch_size
        ), f"RemoteBuffer size {size} must >= batch_size {batch_size}"
        self.name, self.num_workers = name, num_workers
        self.batch_size = batch_size
        self.sparsity = sparsity
        self.buffer_workers = None
        self.subscribe_task = None
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.buffer = Buffer.options(
            name=name, get_if_exists=True, scheduling_strategy=scheduling_local
        ).remote(size, batch_size)
        self.workers = ray.get(self.buffer.get_workers.remote(self.sparsity))
        self.worker_index = random.randint(0, 100)

    def __del__(self):
        self.subscribe_task.cancel() if self.subscribe_task else None

    async def subscribe_workers(cls, buffer, workers, sparsity):
        # 定义为类方法，避免引用self阻止RemoteBuffer被回收
        while True:
            workers[:] = await buffer.subscribe_workers.remote(sparsity)

    async def _init_subscribe_task(self, drop):
        if len(self.workers) != 0 and self.subscribe_task:
            return
        self.worker_index = random.randint(0, 100)
        while not drop and len(self.workers) == 0:
            await self.sync()
            await asyncio.sleep(1)

        if self.subscribe_task:
            await asyncio.sleep(10)
            assert self.workers, f"No buffer worker for {self.name}"
        self.subscribe_task = asyncio.create_task(
            RemoteBuffer.subscribe_workers(
                RemoteBuffer, self.buffer, self.workers, self.sparsity
            )
        )
        self.subscribe_task.add_done_callback(
            lambda t: None if t.cancelled() else t.result()
        )
        await self.sync()
        await self._init_subscribe_task(drop)

    async def __push(self, replays, drop, index):
        num = len(replays)
        if batch := num if self.batch_size and num > 1 else None:
            replays = [make_batch(replays)]
        index = index % len(self.workers)
        while (
            await self.workers[index].push.remote(*replays, drop=drop, batch=batch)
            is False
        ):
            await asyncio.sleep(1)
            index = (index + 1) % len(self.workers)

    async def _push(self, *replays: NestedArray, drop=True) -> None:
        if len(self.workers) == 0 or not self.subscribe_task:
            await self._init_subscribe_task(drop)

        workers_num = len(self.workers)
        if self.batch_size is None:
            step = max(len(replays) // workers_num, 16)
        else:
            step = self.batch_size
        tasks = [
            self.__push(
                replays[i : i + step],
                drop,
                (self.worker_index + i) % workers_num,
            )
            for i in range(0, len(replays), step)
        ]
        self.worker_index += len(tasks)
        try:
            await asyncio.gather(*tasks)
        except ray.exceptions.RayActorError:
            print("Buffer worker is not health, try to sync buffer")
            await self.sync()

    def push(self, *replays: NestedArray) -> asyncio.Task:
        """
        将一个或多个数据推送到 Buffer 中，这些数据会被异步的推送到 Buffer 中，
        当存在多个 BufferWorker 时，会将数据均匀的分配到每个 BufferWorker 中，
        这个方法只能在 asyncio 环境中调用
        Args:
            replays: 待推送的一个或多个数据
        """
        return asyncio.create_task(self._push(*replays, drop=True))

    async def sync(self):
        self.workers[:] = await self.buffer.get_workers.remote(self.sparsity)

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
        ).remote(self.name)

    def __next__(self) -> NestedArray:
        """
        从 Buffer 中读取一条数据，数据的格式为 NestedArray，也就是 push 时的数据格式，
        如果设置了 batch_size，则会将多条数据合并为一个 NestedArray
        """
        if not self.buffer_workers:
            self.buffer_workers = [
                self._new_local_worker() for _ in range(self.num_workers)
            ]
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

        if size != 0:
            return self.replays[self.pop_index]

        for w in self.buffer_workers:
            if w.__bray_ref not in self.ready_replays:
                continue
            ref = w.__bray_ref = w.pop.remote()
            self.next_replays.append(ref)

        pop_beg = time.time()
        self.ready_replays, self.next_replays = ray.wait(
            self.next_replays,
            num_returns=self.num_workers,
            timeout=0,
        )
        if not self.ready_replays:
            self.ready_replays, self.next_replays = ray.wait(
                self.next_replays, num_returns=1
            )
        merge_time_ms("pop_wait", pop_beg, buffer=self.name)
        self.replays.clear()
        for replays in self.ready_replays:
            self.replays.extend(ray.get(replays))
        self.pop_index = 0
        self.last_size += len(self.replays)

        return self.replays[self.pop_index]

    def __iter__(self) -> Iterator[NestedArray]:
        """RemoteBuffer 是一个可迭代对象，可以读取 Buffer 中的数据"""
        return self

    async def _generate(self, source: Iterator[NestedArray]):
        batch_data, batch_size = [], self.batch_size or 1
        last_push_task = asyncio.create_task(asyncio.sleep(0))
        gen_beg = time.time()
        self.worker_index = random.randint(0, 100)
        for data in source:
            merge_time_ms("generate", gen_beg, buffer=self.name)
            batch_data.append(data)
            await asyncio.sleep(0)
            gen_beg = time.time()
            if len(batch_data) < batch_size:
                continue
            push_task = asyncio.create_task(
                self._push(*batch_data, drop=False),
            )
            batch_data.clear()
            await last_push_task
            last_push_task, gen_beg = push_task, time.time()
        await last_push_task
        await self._push(*batch_data, drop=False)

    def _generate_epoch(self, sources, num_workers, epoch):
        num_workers = min(len(sources), num_workers)
        index, workers = num_workers - 1, []

        @ray.remote(num_cpus=0, scheduling_strategy="SPREAD")
        def SourceWorker(source):
            asyncio.run(self._generate(source))

        for i in range(num_workers):
            workers.append(SourceWorker.remote(sources[i]))
            time.sleep(0.5)

        while (index := index + 1) < len(sources):
            rets, workers = ray.wait(workers)
            workers.append(SourceWorker.remote(sources[index]))
            if ret := ray.get(rets[0]) is None:
                continue
            print("SourceWorker is error: ", ret)

        if (epoch := epoch - 1) < 1:
            return
        print(f"Buffer {self.name} remain {epoch} epoch")
        merge("epoch", 1, desc={"cnt": "num epoch"}, buffer=self.name)
        return self._generate_epoch(sources, num_workers, epoch)

    def add_source(self, *sources: Iterator[NestedArray], num_workers=None, epoch=1):
        """
        将一个或多个数据源添加到 Buffer 中，这些数据源会被异步的推送到 Buffer 中

        Args:
            sources: 一个或多个数据源，数据源是一个可迭代对象
            num_workers: 从数据源中读取数据的并发度，默认为 CPU 的核数
            epoch: 从数据源中读取数据的轮数，可以理解为数据源的数据会被重复的读取 epoch 轮

        Returns:
            一个 ray.ObjectRef，可以通过 ray.cancel() 取消数据源的读取
        """
        if num_workers is None:
            cpu_num = sum([node["Resources"]["CPU"] for node in ray.nodes()])
            num_workers = max(1, int(cpu_num))

        @ray.remote(num_cpus=0, scheduling_strategy="SPREAD")
        def RemoteSource(sources, num_workers, epoch):
            self._generate_epoch(sources, num_workers, epoch)

        return RemoteSource.remote(sources, num_workers, epoch)
