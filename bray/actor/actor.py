import ray
import multiprocessing
import time
import asyncio
from asyncio import StreamReader, StreamWriter
import struct
from bray.actor.base import Actor

from bray.metric.metric import (
    merge,
    merge_time_ms,
)


def get_tick_id():
    return 0


class ActorGateway:
    def __init__(self, Actor, args, kwargs, actors_per_worker):
        self.Actor, self.args, self.kwargs = Actor, args, kwargs
        self.actors_per_worker = actors_per_worker
        self.actors = {}
        self.inactive_actors = [
            Actor(*args, **kwargs) for _ in range(actors_per_worker)
        ]
        self.is_initialized = False

    async def _initialize(self):
        self.num_games = 0
        asyncio.create_task(self._check_health())

    async def _check_health(self):
        await asyncio.sleep(60)
        merge(
            "actor",
            len(self.actors),
            desc={"time_window_sum": "smoothed actor num"},
            actor="actor",
        )
        merge(
            "game",
            self.num_games,
            desc={"time_window_sum": "game start per minute"},
            actor="actor",
        )
        self.num_games = 0
        asyncio.create_task(self._check_health())

    def _create_worker(self):
        if len(self.actors) + len(self.inactive_actors) >= self.actors_per_worker:
            raise Exception("Game exceeds max num.")
        return self.Actor(*self.args, **self.kwargs)

    async def _check_active(self, game_id):
        await asyncio.sleep(60)
        actor = self.actors.get(game_id, None)
        if not actor:
            return
        if time.time() - actor.__bray_active_time < 60:
            asyncio.create_task(self._check_active(game_id))
            return
        self.actors.pop(game_id)
        self.inactive_actors.append(actor)
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
            actor.__bray_active_time = time.time()
            start_ret = actor.start(game_id, data)
        except:
            self.actors.pop(game_id, None)
            raise
        asyncio.create_task(self._check_active(game_id))
        self.num_games += 1
        return start_ret

    async def tick(self, game_id, data) -> bytes:
        actor = self.actors.get(game_id, None)
        if not actor:
            raise Exception(f"Game {game_id} not started.")
        try:
            actor.__bray_active_time = time.time()
            tick_ret = await actor.tick(data)
            merge_time_ms("tick", actor.__bray_active_time)
            return tick_ret
        except:
            self.actors.pop(game_id, None)
            raise

    async def end(self, game_id, data) -> bytes:
        actor = self.actors.pop(game_id, None)
        if not actor:
            raise Exception(f"Game {game_id} not started.")
        end_ret = actor.end(data)
        self.inactive_actors.append(actor)
        return end_ret

    async def __call__(self, headers: dict[str:str], body: bytes) -> bytes:
        if not self.is_initialized:
            await self._initialize()
            self.is_initialized = True
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


ACTOR_GATEWAY: ActorGateway = None


def set_actor_gateway(gateway: ActorGateway):
    global ACTOR_GATEWAY
    ACTOR_GATEWAY = gateway


async def handle_client(reader: StreamReader, writer: StreamWriter):
    async def handle(headers, body):
        global ACTOR_GATEWAY
        data = await ACTOR_GATEWAY(headers, body)
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
        except ConnectionResetError:
            print("Client disconnected")
            writer.close()
            return
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


@ray.remote
def serve_gateway(port, Actor, args, kwargs, actors_per_worker):
    async def serve_tcp_gateway():
        server = await asyncio.start_server(
            handle_client, "0.0.0.0", port, reuse_port=True
        )
        async with server:
            await server.serve_forever()

    set_actor_gateway(ActorGateway(Actor, args, kwargs, actors_per_worker))

    asyncio.run(serve_tcp_gateway())


class RemoteActor:
    def __init__(
        self,
        port: int = 8000,
        num_workers: int = 10,
        cpus_per_worker: float = 0.2,
        memory_per_worker: int = 512,
        actors_per_worker: int = 10,
    ):
        """
        Args:
            port: ActorGateway 暴露给 Gamecore 的端口
            num_workers: Actor 的 worker 数量，默认随 Gamecore 的数量自动增长
            cpus_per_worker: 每个 worker 的 CPU 占用量
            memory_per_worker: 每个 worker 的内存占用量，单位 MB
            actors_per_worker: 每个 worker 的 Actor 数量
        """
        self.port, self.num_workers = port, num_workers
        self.cpus_per_worker = cpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.actors_per_worker = actors_per_worker

    def serve(self, Actor: type[Actor], *args, **kwargs):
        """
        Args:
            Actor: 用户定义的 Actor 类
            *args: Actor 的位置参数
            **kwargs: Actor 的关键字参数
        """
        print("Starting ActorGateway.")
        self.gateways = [
            [
                serve_gateway.options(
                    num_cpus=self.num_workers * self.cpus_per_worker,
                    memory=self.num_workers * self.memory_per_worker,
                ).remote(
                    self.port,
                    Actor,
                    args,
                    kwargs,
                    self.actors_per_worker,
                )
                for _ in range(self.num_workers)
            ]
            for node in ray.nodes()
            if node["Alive"]
        ]
        print("ActorGateway started.")
