from fastapi import FastAPI
import ray
from ray import serve

from bray.actor.contract import StepKind, StepRequest, StepRespone


@ray.remote
class ActorWorker:
    def __init__(self, Actor, agents, config, game_id, data):
        self.actor = Actor(agents, config, game_id, data)

    def tick(self, round_id, data):
        return self.actor.tick(round_id, data)

    def end(self, round_id, data):
        return self.actor.end(round_id, data)


# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class ActorGateway:
    def __init__(self, remote_actor):
        self.remote_actor = remote_actor

    # FastAPI will automatically parse the HTTP request for us.
    @app.post("/step")
    async def step(self, req: StepRequest) -> StepRespone:
        return await self.remote_actor.step(req)


class RemoteActor:
    def __init__(self, Actor, agents, config):
        self.Actor = Actor
        self.agents = agents
        self.config = config
        self.workers = {}
        self.actor_gateway = None

    async def _handle_start(self, game_id, data):
        worker = self.workers.get(game_id, None)
        if not worker:
            worker = ActorWorker.remote(
                self.Actor, self.agents, self.config, game_id, data
            )
            self.workers[game_id] = worker
        else:
            raise Exception(f"Game {game_id} already started.")
        return "Game started."

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
    async def _handle_auto(self, game_id, round_id, data):
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

    async def step(self, req: StepRequest) -> StepRespone:
        import base64, pickle

        data = pickle.loads(base64.b64decode(req.data))
        if req.kind == StepKind.start:
            data = await self._handle_start(req.game_id, data)
        elif req.kind == StepKind.tick:
            data = await self._handle_tick(req.game_id, req.round_id, data)
        elif req.kind == StepKind.end:
            data = await self._handle_end(req.game_id, req.round_id, data)
        elif req.kind == StepKind.auto:
            data = await self._handle_auto(req.game_id, req.round_id, data)
        else:
            raise Exception(f"Invalid step kind: {req.kind}")
        data = base64.b64encode(pickle.dumps(data)).decode("utf-8")
        return StepRespone(data=data)

    def serve_background(self):
        actor_gateway = ActorGateway.bind(remote_actor=self)
        print("Starting ActorGateway.")
        serve.run(actor_gateway)
        print("ActorGateway started.")
