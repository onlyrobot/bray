from typing import Iterator, List, Dict
import random
import time

import ray
from bray.utils import ray_scheduling_local
from bray.utils.worker import WorkerLoadBalance
import asyncio

from bray.utils.nested_array import NestedArray, make_batch, split_batch
from bray.master.master import merge, merge_time_ms


@ray.remote(num_cpus=0)
class BufferWorker:
    def __init__(self, name: str):
        self.name, self.buffer = name, ray.get_actor(name)
        self_handle = ray.get_runtime_context().current_actor
        node_id = ray.get_runtime_context().get_node_id()
        worker_info = ray.get(
            self.buffer.register.remote(self_handle, node_id))
        self.size, self.batch_size = worker_info
        self.replays, self._replays, self.out = [], [], []
        self.pop_cond = asyncio.Condition()

    async def push(self, *replays: NestedArray, drop=True, batch=None):
        if not drop:
            return await self._push(replays, drop, batch)
        asyncio.create_task(self._push(replays, drop, batch))

    async def _push(self, replays: NestedArray, drop=True, batch=None):
        if not replays: return  # health check push
        wait_time, interval = 0, 0.01
        while not drop and len(self.replays) >= self.size:
            if wait_time > 1: return False
            wait_time += interval
            await asyncio.sleep(interval)
        before_size = len(self.replays)
        merge(f"push/{self.name}", len(replays) * (batch or 1),
            desc={"time_window_sum": "push per minute"},
        )
        batch_size, _replays = self.batch_size, self._replays
        if batch_size is None or (batch and batch == batch_size):
            self.replays.extend(replays)
        else:
            if batch: replays = split_batch(replays[0])
            _replays.extend(replays)

        while batch_size and len(_replays) >= batch_size:
            out = self.out.pop() if self.out else None
            batch_data = make_batch(_replays[:batch_size], out=out)
            del _replays[:batch_size]
            self.replays.append(batch_data)

        if drop and len(self.replays) > self.size:
            # print(f"Buffer {self.name} is full")
            del self.replays[: -self.size]

        if before_size == 0 and self.replays:
            async with self.pop_cond: self.pop_cond.notify_all()

    async def pop(self) -> List[NestedArray]:
        async with self.pop_cond:
            await self.pop_cond.wait_for(lambda: len(self.replays) > 0)
        replays = self.replays.copy()
        self.replays.clear()  # 重复使用list，避免频繁的内存分配
        batch_size = 1 if self.batch_size is None else self.batch_size
        merge(f"pop/{self.name}", len(replays) * batch_size,
            desc={"time_window_sum": "pop per minute"})
        self.out.extend(replays if self.batch_size else [])
        return replays


@ray.remote(num_cpus=0)
class Buffer:
    def __init__(self, size, batch_size, num_workers, density):
        self.size, self.batch_size = size, batch_size
        self.num_workers, self.density = num_workers, density
        self.worker_cond = asyncio.Condition()
        self.workers = []
        asyncio.create_task(self._health_check())

    async def register(self, worker: BufferWorker, node_id):
        worker.__bray_node_id = node_id
        self.workers.append(worker)
        async with self.worker_cond: self.worker_cond.notify_all()
        return self.size, self.batch_size

    def get_workers(self, name=None, node_id=None) -> list:
        workers = [
            w for w in self.workers if w.__bray_node_id == node_id]
        if not workers: workers = self.workers
        return random.sample(workers, min(self.density, len(workers)))

    async def _is_health(self, worker) -> bool:
        try:
            await worker.push.remote()
            return True
        except ray.exceptions.RayActorError: return False
        except Exception as e:
            print(f"Worker is not health: ", e)
            return False

    async def _health_check(self):
        origin_workers_num = len(self.workers)
        active_workers = [
            worker for worker in self.workers[:origin_workers_num]
            if await self._is_health(worker)
        ]
        old_workers, self.workers = self.workers, active_workers
        self.workers.extend(old_workers[origin_workers_num:])
        await asyncio.sleep(60)
        asyncio.create_task(self._health_check())

    async def subscribe_workers(self, name=None, node_id=None, cur_num=0):
        async with self.worker_cond: 
            await self.worker_cond.wait_for(lambda: cur_num != len(self.workers))
        return self.get_workers(name, node_id)

    async def get_initialize_info(self): 
        return self.batch_size, self.num_workers


class RemoteBuffer:
    remote_buffers: Dict[str, "RemoteBuffer"] = {}

    def __new__(
        cls, name=None, size=512, batch_size=None, num_workers=2, density=100
    ):
        if name in cls.remote_buffers and (
            (self := cls.remote_buffers[name]) is not None
        ):
            return self
        self = super().__new__(cls)
        if name is None: return self    # 适配对象反序列化时调用__new__方法
        self.name, self.buffer_workers = name, None
        self.buffer = Buffer.options(
            name=name, 
            get_if_exists=True, 
            scheduling_strategy=ray_scheduling_local(),
            max_concurrency=100000,
        ).remote(size, batch_size, num_workers, density)
        (
            self.batch_size, self.num_workers
        ) = ray.get(self.buffer.get_initialize_info.remote())
        self.worker_load_balance = WorkerLoadBalance(self.name, self.buffer)
        cls.remote_buffers[name] = self
        return self

    def __init__(self, name: str, *args, **kwargs):
        """
        创建或得到一个 RemoteBuffer，RemoteBuffer 会在 Ray 集群中运行
        Args:
            name: 
        Buffer 的名称，用于标识不同的 Buffer
            size: 
        Buffer 的大小，用于限制 Buffer 中最多能存储多少条数据，
        计算方法为 size * batch_size * num_workers
            batch_size: 
        从 Buffer 中读取数据时，每次读取的数据的批次大小，
        如果为 None，则不对数据进行分批
            num_workers: 
        BufferWorker 的数量，用于控制从 Buffer 中读取数据的并发度
            density: 
        调小后Buffer的生产端和消费端不再是全连接
        """
        assert (
            name in RemoteBuffer.remote_buffers
        ), f"RemoteBuffer {name} is not initialized"

    def __del__(self): self.worker_load_balance.is_initialized = False

    async def __push(self, replays, drop):
        num = len(replays)
        if batch := num if self.batch_size and num > 1 else None:
            replays = [make_batch(replays)]
        retry = 60 if drop else 100000000
        while (await (await self.worker_load_balance.select(
            retry)).push.remote(*replays, drop=drop, batch=batch) is False
        ): await asyncio.sleep(0.01)

    async def _push(self, *replays: NestedArray, drop=True) -> None:
        push_beg = time.time()
        workers_num = len(self.worker_load_balance.workers)
        if not workers_num: workers_num = 1
        if self.batch_size is None:
            step = max(len(replays) // workers_num, 16)
        else:
            step = self.batch_size
        tasks = [self.__push(replays[i : i + step], drop)
            for i in range(0, len(replays), step)]
        try:
            await asyncio.gather(*tasks)
        except ray.exceptions.RayActorError:
            print(f"Buffer {self.name} worker is not health")
        merge_time_ms(f"push/{self.name}", push_beg, mode="remote")

    def push(self, *replays: NestedArray, block=True) -> asyncio.Task:
        """
        将一个或多个数据推送到 Buffer 中，这些数据会被异步的推送到 Buffer 中，
        当存在多个 BufferWorker 时，会将数据均匀的分配到每个 BufferWorker 中，
        这个方法只能在 asyncio 环境中调用
        Args:
            replays: 待推送的一个或多个数据
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(self._push(*replays, drop=True))
        except RuntimeError:
            pass
        from bray.utils import create_or_get_event_loop
    
        return asyncio.run_coroutine_threadsafe(
            self._push(*replays, drop=True), create_or_get_event_loop())

    def _new_local_worker(self) -> BufferWorker:
        """
        创建一个本地的 BufferWorker，这个 BufferWorker 会在当前节点上运行，
        用于从本地的 Buffer 中读取数据，一般被 TrainerWorker 调用
        """
        return BufferWorker.options(
            scheduling_strategy=ray_scheduling_local(), max_concurrency=100000
        ).remote(self.name)

    def __next__(self) -> NestedArray:
        """
        从 Buffer 中读取一条数据，数据的格式为 NestedArray，也就是 push 时的数据格式，
        如果设置了 batch_size，则会将多条数据合并为一个 NestedArray
        """
        if not self.buffer_workers:
            self.buffer_workers = [self._new_local_worker() 
                for _ in range(self.num_workers)]
            self.replays, self.pop_at = [], -1
            self.last_size, self.next_replays = 0, []
        self.pop_at += 1
        remain_size = len(self.replays) - self.pop_at

        # prefetch from buffer when workers num is 1
        if not self.next_replays and remain_size <= self.last_size // 2:
            for w in self.buffer_workers:
                ref = w.__bray_ref = w.pop.remote()
                self.next_replays.append(ref)
            self.last_size, self.ready_replays = 0, []

        if remain_size != 0:
            replay, self.replays[self.pop_at] = (
                self.replays[self.pop_at], None)
            return replay

        for w in self.buffer_workers:
            if w.__bray_ref not in self.ready_replays:
                continue
            ref = w.__bray_ref = w.pop.remote()
            self.next_replays.append(ref)

        pop_beg = time.time()
        self.ready_replays, self.next_replays = ray.wait(
            self.next_replays, 
            num_returns=self.num_workers, timeout=0,
        )
        if not self.ready_replays:
            self.ready_replays, self.next_replays = ray.wait(
                self.next_replays, num_returns=1)
        merge_time_ms(f"wait/{self.name}", pop_beg)
        self.replays.clear()
        for replays in self.ready_replays:
            self.replays.extend(ray.get(replays))
        self.pop_at = 0
        self.last_size += len(self.replays)

        replay, self.replays[0] = self.replays[0], None
        return replay

    def __iter__(self) -> Iterator[NestedArray]:
        """RemoteBuffer 是一个可迭代对象，可以读取 Buffer 中的数据"""
        return self

    async def _generate(self, source: Iterator[NestedArray]):
        batch_data, batch_size = [], self.batch_size or 1
        last_push_task = asyncio.create_task(asyncio.sleep(0))
        gen_beg = time.time()
        self.worker_index = random.randint(0, 100)
        for data in source:
            merge_time_ms(f"generate/{self.name}", gen_beg)
            gen_beg = time.time()
            batch_data.append(data)
            if len(batch_data) < batch_size:
                await asyncio.sleep(0)
                continue
            push_task = asyncio.create_task(
                self._push(*batch_data, drop=False),
            )
            batch_data.clear()
            await last_push_task
            last_push_task, gen_beg = push_task, time.time()
        await last_push_task
        await self._push(*batch_data, drop=False)

    def _generate_epoch(self, sources: List[Iterator], num_workers):
        @ray.remote(num_cpus=0, scheduling_strategy="SPREAD")
        def SourceWorker(source):
            asyncio.run(self._generate(iter(source)))

        num_workers = min(len(sources), num_workers)
        index, workers = num_workers - 1, []

        for i in range(num_workers):
            workers.append(SourceWorker.remote(sources[i]))

        while (index := index + 1) < len(sources):
            rets, workers = ray.wait(workers)
            workers.append(SourceWorker.remote(sources[index]))
            if (ret := ray.get(rets[0])) is None:
                continue
            print("SourceWorker error: ", ret)
        while len(workers) > 0:
            rets, workers = ray.wait(workers)
            if (ret := ray.get(rets[0])) is None:
                continue
            print("SourceWorker error: ", ret)
        merge(f"epoch/{self.name}", 1, desc={"cnt": "num epoch"})

    def add_source(
        self, *sources: Iterator[NestedArray], num_workers=None, epoch=1
    ):
        """
        将一个或多个数据源添加到 Buffer 中，这些数据源会被异步的推送到 Buffer 中

        Args:
            sources: 一个或多个数据源，数据源是一个可迭代对象
            num_workers: 从数据源中读取数据的并发度，默认为 CPU 的核数
            epoch: 从数据源中重复读取数据的轮数，也就是 epoch 数

        Returns:
            一个 ray.ObjectRef，可以通过 ray.cancel() 取消数据源的读取
        """
        if num_workers is None:
            num_workers = max(1, int(ray.available_resources()["CPU"]))

        @ray.remote(num_cpus=0, scheduling_strategy="SPREAD")
        def Source(sources, num_workers, epoch):
            for i in range(epoch):
                print(f"Buffer {self.name} epoch {i} start")
                self._generate_epoch(sources, num_workers)
            print(f"Buffer {self.name} epoch done")

        return Source.remote(sources, num_workers, epoch)
