import ray
from fastapi import FastAPI
import uvicorn
from starlette.requests import Request
from starlette.responses import Response
from threading import Thread
import time
import logging
import asyncio
from bray.actor.base import Actor

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from bray.metric.metric import (
    merge,
    flush_metrics_to_remote,
    merge_time_ms,
)


@ray.remote(max_restarts=1, num_cpus=0.5)
class ActorWorker:
    def __init__(self, Actor, *args, **kwargs):
        self.active_time, self.check_time = time.time(), 0
        self.actor = Actor(*args, **kwargs)
        asyncio.create_task(self._health_check())

    def start(self, game_id, data: bytes) -> bytes:
        self.active_time = time.time()
        return self.actor.start(game_id, data)

    async def tick(self, data: bytes) -> bytes:
        self.active_time = time.time()
        ret = await self.actor.tick(data)
        merge_time_ms("tick", self.active_time)
        return ret

    def end(self, data: bytes) -> bytes:
        return self.actor.end(data)

    async def _health_check(self):
        await asyncio.sleep(60 * 2)
        if time.time() - self.check_time < 60 * 2 or self.is_active():
            asyncio.create_task(self._health_check())
            return
        flush_metrics_to_remote()
        print("ActorGateway inactive, worker exit.")
        ray.kill(ray.get_runtime_context().current_actor)

    def __del__(self):
        flush_metrics_to_remote()

    def is_active(self) -> bool:
        self.check_time = time.time()
        return self.check_time - self.active_time < 60


app = FastAPI()
actor_gateway: "ActorGateway" = None


def serve_actor_gateway(gateway: "ActorGateway"):
    global app, actor_gateway
    actor_gateway = gateway
    Thread(
        target=uvicorn.run, kwargs={"app": app, "log_level": logging.WARNING}
    ).start()


@app.post("/step")
async def step(request: Request):
    global actor_gateway
    return Response(content=await actor_gateway(request))


@ray.remote
class ActorGateway:
    def __init__(self, Actor: type[Actor], args, kwargs, num_workers):
        self.Actor, self.args, self.kwargs = Actor, args, kwargs
        self.num_workers = num_workers
        self.workers = {}
        self.inactive_workers = [
            ActorWorker.remote(self.Actor, *self.args, **self.kwargs)
            for _ in range(num_workers)
        ]
        serve_actor_gateway(self)
        # uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
        # import logging

        # logger = logging.getLogger("ray.serve")
        # logger.setLevel(logging.WARNING)
        asyncio.create_task(self._health_check())

    def _create_worker(self):
        if (
            self.num_workers != 0
            and len(self.workers) + len(self.inactive_workers) >= self.num_workers
        ):
            raise Exception("Game exceeds max num.")
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        return ActorWorker.options(scheduling_strategy=scheduling_local).remote(
            self.Actor, *self.args, **self.kwargs
        )

    async def _health_check(self):
        await asyncio.sleep(60)
        if (
            self.num_workers == 0
            and len(self.inactive_workers) > len(self.workers) // 2
            and len(self.inactive_workers) > 2
        ):
            try:
                self.inactive_workers.pop()
            except IndexError:
                pass
        try:
            await asyncio.gather(
                *[worker.is_active.remote() for worker in self.inactive_workers]
            )
        except Exception as e:
            print(f"Health check failed: {e}")
        asyncio.create_task(self._health_check())

    async def _active_check(self, game_id):
        await asyncio.sleep(60)
        worker = self.workers.get(game_id, None)
        if not worker:
            return
        try:
            if await worker.is_active.remote():
                asyncio.create_task(self._active_check(game_id))
                return
        except Exception as e:
            print(f"Health check failed: {e}")
            self.workers.pop(game_id)
            return
        self.workers.pop(game_id)
        self.inactive_workers.append(worker)
        print(f"Actor with game_id={game_id} inactive.")

    async def start(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if worker:
            raise Exception(f"Game {game_id} already started.")
        try:
            worker = self.inactive_workers.pop()
        except IndexError:
            worker = self._create_worker()
        self.workers[game_id] = worker
        merge(
            "worker",
            len(self.workers),
            desc={
                "time_window_avg": "smoothed actor worker num",
                "time_window_cnt": "game start per minute",
            },
            actor="actor",
        )
        try:
            start_ret = await worker.start.remote(game_id, data)
        except:
            self.workers.pop(game_id, None)
            raise
        asyncio.create_task(self._active_check(game_id))
        return start_ret

    async def tick(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        try:
            return await worker.tick.remote(data)
        except:
            self.workers.pop(game_id, None)
            raise

    async def end(self, game_id, data) -> bytes:
        worker = self.workers.pop(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        end_ret = await worker.end.remote(data)
        self.inactive_workers.append(worker)
        return end_ret

    async def __call__(self, req: Request) -> bytes:
        step_kind = req.headers.get("step_kind")
        game_id = req.headers.get("game_id")
        if game_id is None:
            raise Exception("game_id must be provided.")
        data = await req.body()
        if step_kind == "tick":
            return await self.tick(game_id, data)
        elif step_kind == "start":
            return await self.start(game_id, data)
        elif step_kind == "end":
            return await self.end(game_id, data)
        raise Exception("Unknown step_kind:", step_kind)


class RemoteActor:
    def __init__(self, port: int = 8000, num_workers: int = 0):
        """
        Args:
            port: ActorGateway 暴露给 Gamecore 的端口
            num_workers: Actor 的 worker 数量，默认随 Gamecore 的数量自动增长
        """
        self.port, self.num_workers = port, num_workers

    def serve(self, Actor: type[Actor], *args, **kwargs):
        """
        Args:
            Actor: 用户定义的 Actor 类
            *args: Actor 的位置参数
            **kwargs: Actor 的关键字参数
        """
        # total_cpus = ray.available_resources()["CPU"]
        # num_replicas = total_cpus // 16 + 1
        # self.gateway = ActorGateway.options(num_replicas=num_replicas).bind(
        #     Actor, args, kwargs, self.num_workers
        # )
        # self.gateway = ActorGateway.options(is_driver_deployment=True).bind(
        #     Actor, args, kwargs, self.num_workers
        # )
        print("Starting ActorGateway.")

        self.gateways = [
            ActorGateway.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )
            ).remote(Actor, args, kwargs, self.num_workers)
            for node in ray.nodes()
        ]
        # self.gateway.serve.remote()
        # serve.shutdown()
        # serve.run(self.gateway, host="0.0.0.0", port=self.port)
        print("ActorGateway started.")
