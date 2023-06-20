from fastapi import FastAPI
import ray
from ray import serve
from starlette.requests import Request
import time
import asyncio
from bray.actor.base import Actor
from bray.actor.base import Agent

from enum import Enum


class StepKind(str, Enum):
    start = "start"
    tick = "tick"
    end = "end"
    auto = "auto"


@ray.remote
class ActorWorker:
    def __init__(self, Actor, agents, config, game_id, data):
        self.active_time = time.time()
        self.actor = Actor(agents, config, game_id, data)

    def tick(self, round_id, data) -> bytes:
        self.active_time = time.time()
        return self.actor.tick(round_id, data)

    def end(self, round_id, data) -> bytes:
        return self.actor.end(round_id, data)

    async def is_active(self) -> bool:
        return time.time() - self.active_time < 60 * 3


# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class ActorGateway:
    def __init__(self, remote_actor: "RemoteActor"):
        self.remote_actor = remote_actor

    # FastAPI will automatically parse the HTTP request for us.
    @app.post("/step")
    async def step(self, req: Request) -> bytes:
        round_id = int(req.headers.get("round_id"))
        game_id = req.headers.get("game_id")
        if game_id is None or round_id is None:
            raise Exception("game_id and round_id must be provided.")
        step_kind = StepKind(req.headers.get("step_kind"))
        return await self.remote_actor.step(
            game_id, round_id, step_kind, await req.body()
        )


class RemoteActor:
    def __init__(self, Actor: Actor, agents: dict[str:Agent], config: any):
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

    async def _handle_start(self, game_id, data):
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

    async def _handle_tick(self, game_id, round_id, data):
        worker = self.workers.get(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        return await worker.tick.remote(round_id, data)

    async def _handle_end(self, game_id, round_id, data):
        worker = self.workers.pop(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        return await worker.end.remote(round_id, data)

    # for stateless actors, we can just use the actor class directly.
    async def _handle_auto(self, round_id, data):
        try:
            worker = self.workers.popitem()
        except KeyError:
            worker = ActorWorker.remote(
                self.Actor, self.agents, self.config, None, None
            )
        data = await worker.tick.remote(round_id, data)
        import uuid

        self.workers[uuid.uuid4().hex] = worker
        return data

    async def step(
        self, game_id: str, round_id: int, step_kind: StepKind, data: bytes
    ) -> bytes:
        if step_kind == StepKind.start:
            return await self._handle_start(game_id, data)
        elif step_kind == StepKind.tick:
            return await self._handle_tick(game_id, round_id, data)
        elif step_kind == StepKind.end:
            return await self._handle_end(game_id, round_id, data)
        elif step_kind == StepKind.auto:
            return await self._handle_auto(round_id, data)
        else:
            raise Exception(f"Invalid step kind: {step_kind}")

    def serve_background(self, port=8000):
        actor_gateway = ActorGateway.bind(remote_actor=self)
        print("Starting ActorGateway.")
        serve.run(actor_gateway, port=port)
        print("ActorGateway started.")
