import ray
from ray import serve
from starlette.requests import Request
import time
import asyncio
from bray.actor.base import Actor

from bray.metric.metric import merge, flush_metrics_to_remote


@ray.remote
class ActorWorker:
    def __init__(self, Actor, *args, **kwargs):
        self.active_time = time.time()
        self.actor = Actor(*args, **kwargs)

    def start(self, game_id, data: bytes) -> bytes:
        merge("game", 1, desc={"time_window_cnt": "game start per minute"})
        self.active_time = time.time()
        return self.actor.start(game_id, data)

    def tick(self, data: bytes) -> bytes:
        merge("tick", 1, desc={"time_window_cnt": "game tick per minute"})
        self.active_time = time.time()
        return self.actor.tick(data)

    def end(self, data: bytes) -> bytes:
        return self.actor.end(data)

    def __del__(self):
        flush_metrics_to_remote()

    def is_active(self) -> bool:
        return time.time() - self.active_time < 60


@serve.deployment(route_prefix="/step")
class ActorGateway:
    def __init__(self, Actor: type[Actor], *args, **kwargs):
        self.Actor, self.args, self.kwargs = Actor, args, kwargs
        self.workers = {}
        import logging

        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.WARNING)

    async def _check_workers(self, game_id):
        worker = self.workers.get(game_id, None)
        if not worker:
            return
        is_active = await worker.is_active.remote()
        if not is_active:
            print(f"Actor with game_id={game_id} inactive.")
            self.workers.pop(game_id)
            return
        await asyncio.sleep(60)
        asyncio.create_task(self._check_workers(game_id))

    async def __call__(self, req: Request) -> bytes:
        step_kind = req.headers.get("step_kind")
        game_id = req.headers.get("game_id")
        if game_id is None:
            raise Exception("game_id must be provided.")
        data = await req.body()
        if step_kind == "start":
            return await self.start(game_id, data)
        elif step_kind == "tick":
            return await self.tick(game_id, data)
        elif step_kind == "end":
            return await self.end(game_id, data)
        elif step_kind == "auto":
            return await self.auto(game_id, data)
        raise Exception(f"Unknown step_kind: {step_kind}")

    async def start(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if worker:
            raise Exception(f"Game {game_id} already started.")
        worker = ActorWorker.remote(self.Actor, *self.args, **self.kwargs)
        worker.start.remote(game_id, data)
        self.workers[game_id] = worker
        merge(
            "worker",
            len(self.workers),
            desc={"time_window_avg": "smoothed actor worker num"},
            actor="actor",
        )
        asyncio.create_task(self._check_workers(game_id))
        return b"Game started."

    async def tick(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        return await worker.tick.remote(data)

    async def end(self, game_id, data) -> bytes:
        worker = self.workers.pop(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        return await worker.end.remote(data)

    # for stateless actors, we can just use the actor class directly.
    async def auto(self, data: bytes) -> bytes:
        try:
            worker = self.workers.popitem()
        except KeyError:
            worker = ActorWorker.remote(self.Actor, *self.args, **self.kwargs)
            worker.start(None, None)
        data = await worker.tick.remote(data)
        import uuid

        self.workers[uuid.uuid4().hex] = worker
        return data


class RemoteActor:
    def __init__(self, port: int = 8000):
        """
        Args:
            port: ActorGateway 暴露给 Gamecore 的端口
        """
        self.port = port

    def serve(self, Actor: type[Actor], *args, **kwargs):
        """
        Args:
            Actor: 用户定义的 Actor 类
            *args: Actor 的位置参数
            **kwargs: Actor 的关键字参数
        """
        self.gateway = ActorGateway.bind(Actor, *args, **kwargs)
        print("Starting ActorGateway.")
        serve.run(self.gateway, host="0.0.0.0", port=self.port)
        print("ActorGateway started.")
