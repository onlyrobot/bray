import ray
from fastapi import FastAPI
import uvicorn
from starlette.requests import Request
from starlette.responses import Response
from threading import Thread
import time
import logging
import asyncio
import struct
from bray.actor.base import Actor

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from bray.metric.metric import (
    merge,
    flush_metrics_to_remote,
    merge_time_ms,
)


TICK_ID: int = 0


def set_tick_id(tick_id: int):
    global TICK_ID
    TICK_ID = tick_id


def get_tick_id() -> int:
    return TICK_ID


@ray.remote(max_restarts=1)
class ActorWorker:
    def __init__(self, Actor, *args, **kwargs):
        self.active_time, self.check_time = time.time(), 0
        self.actor = Actor(*args, **kwargs)
        self.need_set_tick_id = len(ray.nodes()) == 1
        asyncio.create_task(self._health_check())

    def start(self, game_id, data: bytes) -> bytes:
        self.active_time = time.time()
        return self.actor.start(game_id, data)

    async def tick(self, tick_id: int, data: bytes) -> bytes:
        set_tick_id(tick_id if self.need_set_tick_id else 0)
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


def serve_http_gateway(gateway: "ActorGateway"):
    global app, actor_gateway
    actor_gateway = gateway
    Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "0.0.0.0", "log_level": logging.WARNING},
    ).start()


@app.post("/step")
async def step(request: Request):
    global actor_gateway
    headers, body = request.headers, await request.body()
    return Response(content=await actor_gateway(headers, body))


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    global actor_gateway

    async def handle(headers, body):
        data = await actor_gateway(headers, body)
        # 计算Header中的字段值
        game_id_size = len(headers["game_id"])
        body_size = len(data)
        time = headers["time"]
        # 构造Header
        header = struct.pack("!3q", game_id_size, body_size, time)
        # 构造整个包
        try:
            writer.write(header + headers["game_id"] + data)
            await writer.drain()
        except Exception as e:
            print(e)
            writer.close()

    while True:
        # 接收客户端请求数据
        try:
            data = await reader.readexactly(8 * 6)
            (
                game_id_size,
                step_kind_size,
                key_size,
                token_size,
                body_size,
                time,
            ) = struct.unpack("!6q", data)
            data = await reader.readexactly(
                game_id_size + step_kind_size + key_size + token_size + body_size
            )
        except Exception as e:
            print(e)
            writer.close()
            return
        game_id = data[0:game_id_size]
        step_kind = data[game_id_size : game_id_size + step_kind_size]
        offset = game_id_size + step_kind_size
        key = data[offset : offset + key_size]
        offset += key_size
        token = data[offset : offset + token_size]
        body = data[offset + token_size :]
        headers = {
            "game_id": game_id,
            "step_kind": step_kind.decode(),
            "key": key.decode(),
            "token": token.decode(),
            "time": time,
        }
        asyncio.create_task(handle(headers, body))


async def serve_tcp_gateway(gateway: "ActorGateway"):
    global actor_gateway
    actor_gateway = gateway
    # 创建TCP服务器
    server = await asyncio.start_server(handle_client, "0.0.0.0", 8000)

    # 开始监听端口
    async with server:
        await server.serve_forever()


@ray.remote
class ActorGateway:
    def __init__(
        self,
        Actor: type[Actor],
        args: tuple[any],
        kwargs: dict[str, any],
        num_workers: int,
        cpus_per_worker: float,
        memory_per_worker: int,
        actors_per_worker: int,
        use_tcp: bool,
    ):
        self.Actor, self.args, self.kwargs = Actor, args, kwargs
        self.num_workers = num_workers if num_workers else 0
        self.workers = {}
        self.tick_id, self.num_games = 0, 0
        scheduling_local = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        self.RemoteActorWorker = ActorWorker.options(
            num_cpus=cpus_per_worker,
            memory=memory_per_worker * 1024 * 1024,
            scheduling_strategy=scheduling_local,
        )
        self.inactive_workers = [
            self.RemoteActorWorker.remote(self.Actor, *self.args, **self.kwargs)
            for _ in range(self.num_workers)
        ]
        if not use_tcp:
            serve_http_gateway(self)
        else:
            asyncio.create_task(serve_tcp_gateway(self))
        # uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")
        # import logging

        # logger = logging.getLogger("ray.serve")
        # logger.setLevel(logging.WARNING)
        asyncio.create_task(self._health_check())

    async def _create_worker(self):
        if (
            self.num_workers != 0
            and len(self.workers) + len(self.inactive_workers) >= self.num_workers
        ):
            raise Exception("Game exceeds max num.")
        return self.RemoteActorWorker.remote(self.Actor, *self.args, **self.kwargs)

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
        merge(
            "worker",
            len(self.workers),
            desc={"time_window_sum": "smoothed actor worker num"},
            actor="actor",
        )
        merge(
            "game",
            self.num_games,
            desc={"time_window_sum": "game start per minute"},
            actor="actor",
        )
        self.num_games = 0
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
            worker = await self._create_worker()
        try:
            start_ret = await worker.start.remote(game_id, data)
            self.workers[game_id] = worker
        except:
            raise
        self.num_games += 1
        asyncio.create_task(self._active_check(game_id))
        return start_ret

    async def tick(self, game_id, data) -> bytes:
        worker = self.workers.get(game_id, None)
        if not worker:
            raise Exception(f"Game {game_id} not started.")
        try:
            self.tick_id += 1
            return await worker.tick.remote(self.tick_id, data)
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

    async def __call__(self, headers: dict[str:str], body: bytes) -> bytes:
        step_kind = headers.get("step_kind")
        game_id = headers.get("game_id")
        if game_id is None:
            raise Exception("game_id must be provided.")
        if step_kind == "tick":
            return await self.tick(game_id, body)
        elif step_kind == "start":
            return await self.start(game_id, body)
        elif step_kind == "end":
            return await self.end(game_id, body)
        raise Exception("Unknown step_kind:", step_kind)


class RemoteActor:
    def __init__(
        self,
        port: int = 8000,
        num_workers: int = None,
        cpus_per_worker: float = 0.2,
        memory_per_worker: int = 512,
        actors_per_worker: int = 1,
        use_tcp: bool = False,
    ):
        """
        Args:
            port: ActorGateway 暴露给 Gamecore 的端口
            num_workers: Actor 的 worker 数量，默认随 Gamecore 的数量自动增长
            cpus_per_worker: 每个 worker 的 CPU 占用量
            memory_per_worker: 每个 worker 的内存占用量，单位 MB
            actors_per_worker: 每个 worker 的 Actor 数量
            use_tcp: 是否使用 TCP 协议
        """
        self.port, self.num_workers = port, num_workers
        self.cpus_per_worker = cpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.actors_per_worker = actors_per_worker
        self.use_tcp = use_tcp

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
                ),
            ).remote(
                Actor,
                args,
                kwargs,
                self.num_workers,
                self.cpus_per_worker,
                self.memory_per_worker,
                self.actors_per_worker,
                self.use_tcp,
            )
            for node in ray.nodes()
            if node["Alive"]
        ]
        # self.gateway.serve.remote()
        # serve.shutdown()
        # serve.run(self.gateway, host="0.0.0.0", port=self.port)
        print("ActorGateway started.")
