import asyncio, ray, random

class WorkerLoadBalance:
    def __init__(self, name, service):
        """服务发现和Worker负载均衡，指定的service需要满足接口要求"""
        self.name, self.service = name, service
        self.node_id = ray.get_runtime_context().get_node_id()
        self.workers = ray.get(self.get_workers())
        self.worker_index = random.randint(0, 100000)
        self.is_initialized = False

    async def select(self, retry=60) -> "Worker":
        if not self.workers or not self.is_initialized:
            await self.initialize(retry)
        self.worker_index += 1
        return self.workers[self.worker_index % len(self.workers)]

    async def initialize(self, retry):
        if self.workers and self.is_initialized: return
        self.worker_index = random.randint(0, 100000)
        self.node_id = ray.get_runtime_context().get_node_id()
        if not self.is_initialized:
            asyncio.create_task(self.subscribe_workers())
            self.is_initialized = True
        retry, interval = retry, 1
        while not self.workers and (retry := retry - 1):
            await asyncio.sleep(interval)
        assert self.workers, f"No worker for {self.name}"
        self.workers = await self.get_workers()

    async def subscribe_workers(self):
        coro = self.service.subscribe_workers.remote(
            self.name, self.node_id, len(self.workers))
        try:
            self.workers = await asyncio.wait_for(
                coro, timeout=10 * 60,
            )
        except asyncio.TimeoutError: pass
        except Exception as e:
            await asyncio.sleep(0.1)
            print(f"Fail to subscribe workers for {self.name}")
        if not self.is_initialized: return
        asyncio.create_task(self.subscribe_workers())

    def get_workers(self) -> ray.ObjectRef:
        return self.service.get_workers.remote(self.name, self.node_id)