import ray
from ray import serve
from starlette.requests import Request
import time
import asyncio
from bray.actor.base import Actor
from bray.actor.base import Agent


@ray.remote
class ActorWorker:
    def __init__(self, Actor, agents, config, game_id, data):
        self.active_time = time.time()
        self.actor = Actor(agents, config, game_id, data)

    def tick(self, data: bytes) -> bytes:
        self.active_time = time.time()
        return self.actor.tick(data)

    def end(self, data: bytes) -> bytes:
        return self.actor.end(data)

    async def is_active(self) -> bool:
        return time.time() - self.active_time < 60


@serve.deployment(route_prefix="/step")
class _RemoteActor:
    def __init__(self, Actor: type[Actor], agents: dict[str:Agent], config: any):
        self.Actor, self.agents, self.config = Actor, agents, config
        self.workers = {}

    async def _check_workers(self, game_id):
        worker = self.workers.get(game_id, None)
        if not worker:
            return
        is_active = await worker.is_active.remote()
        if not is_active:
            print(f"Worker with game_id={game_id} inactive, shutting down.")
            self.workers.pop(game_id)
            return
        await asyncio.sleep(60)
        await asyncio.create_task(self._check_workers(game_id))

    async def _parse_request(self, req: Request):
        game_id = req.headers.get("game_id")
        if game_id is None:
            raise Exception("game_id must be provided.")
        data = await req.body()
        return game_id, data

    async def __call__(self, req: Request) -> bytes:
        step_kind = req.headers.get("step_kind")
        game_id, data = await self._parse_request(req)
        if step_kind == "start":
            return await self.start(game_id, data)
        elif step_kind == "tick":
            return await self.tick(game_id, data)
        elif step_kind == "end":
            return await self.end(game_id, data)
        elif step_kind == "auto":
            return await self.auto(game_id, data)
        else:
            raise Exception(f"Unknown step_kind: {step_kind}")

    async def start(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if not worker:
            worker = ActorWorker.remote(
                self.Actor, self.agents, self.config, game_id, data
            )
            self.workers[game_id] = worker
            asyncio.create_task(self._check_workers(game_id))
        else:
            raise Exception(f"Game {game_id} already started.")
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
            worker = ActorWorker.remote(
                self.Actor, self.agents, self.config, None, None
            )
        data = await worker.tick.remote(data)
        import uuid

        self.workers[uuid.uuid4().hex] = worker
        return data


class RemoteActor:
    def __init__(
        self,
        port: int,
        Actor: type[Actor],
        agents: dict[str:Agent] = None,
        config: any = None,
    ):
        print("Starting ActorGateway.")
        self._remote_actor = _RemoteActor.bind(Actor, agents, config)
        serve.run(self._remote_actor, port=port)
        print("ActorGateway started.")
