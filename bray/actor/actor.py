import ray
import time
import asyncio
import traceback
from asyncio import StreamReader, StreamWriter
import struct
from bray.actor.base import Actor
from bray.actor.gateway import Gateway

from bray.metric.metric import (
    merge,
    merge_time_ms,
)


class ActorGateway:
    def __init__(self, name, Actor, args, kwargs, actors_per_worker):
        self.name = name
        self.Actor, self.args, self.kwargs = Actor, args, kwargs
        self.actors_per_worker = actors_per_worker
        self.actors = {}
        self.auto_actor = None
        self.concurrency = 0
        self.inactive_actors = [
            Actor(*args, **kwargs) for _ in range(actors_per_worker)
        ]
        self.is_initialized = False
        self.active_check_interval = 60

    def _initialize(self):
        if self.is_initialized:
            return
        self.num_games = 0
        asyncio.create_task(self._check_health())
        self.is_initialized = True

    async def _check_health(self):
        await asyncio.sleep(60)
        merge(
            f"actor/{self.name}",
            len(self.actors),
            desc={"time_window_sum": "smoothed actor num"},
        )
        merge(
            f"game/{self.name}",
            self.num_games,
            desc={"time_window_sum": "game start per minute"},
        )
        self.num_games = 0
        asyncio.create_task(self._check_health())

    def _create_worker(self):
        if len(self.actors) >= self.actors_per_worker:
            raise Exception("Game exceeds max num.")
        return self.Actor(*self.args, **self.kwargs)

    async def _check_active(self, game_id):
        await asyncio.sleep(self.active_check_interval)
        actor = self.actors.get(game_id, None)
        if not actor:
            return
        interval = time.time() - actor.__bray_atime
        if interval < self.active_check_interval:
            asyncio.create_task(self._check_active(game_id))
            return
        # self.inactive_actors.append(actor)
        self.actors.pop(game_id)
        print(f"Actor with game_id={game_id} inactive.")

    async def start(self, game_id, data) -> bytes:
        actor = self.actors.get(game_id, None)
        if actor:
            raise Exception(f"Game {game_id} already started.")
        try:
            actor = self.inactive_actors.pop()
        except IndexError:
            actor = self._create_worker()
        self.actors[game_id] = actor
        try:
            actor.__bray_atime = time.time()
            start_ret = await actor.start(game_id, data)
        except:
            self.actors.pop(game_id, None)
            raise
        self.num_games += 1
        asyncio.create_task(self._check_active(game_id))
        return start_ret

    async def tick(self, game_id, data) -> bytes:
        actor = self.actors.get(game_id, None)
        if not actor:
            raise Exception(f"Game {game_id} not started.")
        actor.__bray_atime = time.time()
        try:
            tick_ret = await actor.tick(data)
        except:
            self.actors.pop(game_id, None)
            raise
        merge_time_ms(f"tick/{self.name}", actor.__bray_atime)
        return tick_ret

    async def auto(self, data) -> bytes:
        beg = time.time()
        while self.concurrency >= self.actors_per_worker * 5:
            await asyncio.sleep(0.001)
        if self.auto_actor is None:
            try:
                actor = self.inactive_actors.pop()
            except IndexError:
                actor = self._create_worker()
            self.auto_actor = actor
        try:
            self.concurrency += 1
            tick_ret = await self.auto_actor.tick(data)
            self.concurrency -= 1
        except:
            self.auto_actor = None
            raise
        merge_time_ms(f"tick/{self.name}", beg)
        return tick_ret

    async def __call__(self, headers: dict, body: bytes) -> bytes:
        if not self.is_initialized:
            self._initialize()
        step_kind = headers.get("step_kind", "auto")

        if step_kind == "auto":
            return await self.auto(body)

        game_id = headers.get("game_id")
        if game_id is None:
            raise Exception("game_id must be provided.")

        if step_kind == "tick":
            return await self.tick(game_id, body)

        if step_kind == "start":
            return await self.start(game_id, body)

        if step_kind != "stop":
            raise Exception("Unknown step_kind:", step_kind)

        actor = self.actors.pop(game_id, None)
        if not actor:
            raise Exception(f"Game {game_id} not started.")

        stop_ret = await actor.stop(body)
        self.inactive_actors.append(actor)
        return stop_ret


ACTOR_GATEWAY: ActorGateway = None


def set_actor_gateway(gateway: ActorGateway):
    global ACTOR_GATEWAY
    ACTOR_GATEWAY = gateway


async def handle_client(reader: StreamReader, writer: StreamWriter):
    async def handle(headers, body):
        global ACTOR_GATEWAY
        try:
            data = await ACTOR_GATEWAY(headers, body)
        except:
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()
            return
        if not isinstance(data, bytes):
            raise Exception("Actor return must be bytes")
        game_id_size = len(headers["game_id"])
        body_size = len(data)
        time = headers["time"]
        try:
            header = struct.pack("!3q", game_id_size, body_size, time)
            writer.write(header + headers["game_id"] + data)
            await writer.drain()
        except:
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()

    while True:
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
        except (
            ConnectionResetError,
            asyncio.exceptions.IncompleteReadError,
        ):
            print("Client disconnected")
            writer.close()
            await writer.wait_closed()
            return
        except:
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()
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


@ray.remote(max_retries=-1)
def ActorWorker(name, port, Actor, args, kwargs, actors_per_worker, use_tcp, gateway):
    gateway.register.remote("localhost", port) if gateway else None

    async def serve_tcp_gateway():
        server = await asyncio.start_server(
            handle_client, "0.0.0.0", port, reuse_port=True
        )
        async with server:
            await server.serve_forever()

    try:
        gateway = ActorGateway(name, Actor, args, kwargs, actors_per_worker)
        set_actor_gateway(gateway)
    except:
        traceback.print_exc()
        raise

    if use_tcp:
        return asyncio.run(serve_tcp_gateway())

    from fastapi import FastAPI, HTTPException
    import uvicorn, socket
    from starlette.requests import Request
    from starlette.responses import Response
    from logging import WARN

    async def step(request: Request):
        headers, body = request.headers, await request.body()
        try:
            data = await gateway(headers, body)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=data)

    app = FastAPI(docs_url=None, redoc_url=None)
    app.add_api_route("/step", step, methods=["POST"])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind(("0.0.0.0", port))
    uvicorn.run(app, fd=sock.fileno(), timeout_keep_alive=60 * 5, log_level=WARN)


class RemoteActor:
    def __init__(
        self,
        name: str = "default",
        port: int = 8000,
        num_workers: int = 2,
        cpus_per_worker: float = 1,
        memory_per_worker: int = 512,
        actors_per_worker: int = 10,
        use_tcp: bool = False,
        use_gateway: ["node", "head", None] = "node",
    ):
        """
        Args:
            port: ActorGateway 暴露给 Gamecore 的端口
            num_workers: RemoteActor 的 worker 数量，将会在每个节点启动 num_workers 个 worker
            cpus_per_worker: 每个 worker 的 CPU 占用量
            memory_per_worker: 每个 worker 的内存占用量，单位 MB
            actors_per_worker: 每个 worker 的 Actor 数量，总 Actor 数量为 num_workers * actors_per_worker
            use_tcp: 是否使用 TCP 作为通信协议
            gateway: ActorGateway 的位置，可以是 "node" 或 "head" 或 None
        """
        self.name, self.port = name, port
        self.num_workers = num_workers
        self.cpus_per_worker = cpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.actors_per_worker = actors_per_worker
        self.use_tcp = use_tcp
        self.use_gateway = use_gateway

    def serve(self, Actor: type[Actor], *args, **kwargs):
        """
        Args:
            Actor: 用户定义的 Actor 类
            *args: Actor 的位置参数
            **kwargs: Actor 的关键字参数
        """
        print(f"Starting Actor {self.name}...")
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        self.node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
        gateway = (
            None
            if self.use_gateway != "head"
            else Gateway.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            ).remote(self.port)
        )
        self.gateways = [
            gateway
            if gateway or not self.use_gateway
            else Gateway.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(self.port)
            for node_id in self.node_ids
        ]
        self.workers = [
            ActorWorker.options(
                num_cpus=self.cpus_per_worker,
                memory=self.memory_per_worker * 1024 * 1024,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            ).remote(
                self.name,
                self.port + (i + 1 if self.use_gateway else 0),
                Actor,
                args,
                kwargs,
                self.actors_per_worker,
                self.use_tcp,
                gateway,
            )
            for node_id, gateway in zip(self.node_ids, self.gateways)
            for i in range(self.num_workers)
        ]
        ray.get([gateway.serve.remote() for gateway in self.gateways])
        print(f"Actor {self.name} started at port {self.port}.")
