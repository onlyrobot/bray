from asyncio import StreamReader, StreamWriter
import asyncio
from typing import Type, Tuple, Dict
import time
import struct
import traceback
import random

import ray
from ray import get_runtime_context
from bray.master.master import (
    merge,
    merge_time_ms,
)
from bray.utils import ray_scheduling_local
from bray.server.base import Server
from bray.server.gateway import Gateway


class ServerGateway:
    def __init__(self, name, Server, args, kwargs, servers_per_worker):
        self.name = name
        self.Server, self.args, self.kwargs = Server, args, kwargs
        self.servers_per_worker = servers_per_worker
        self.servers = {}
        self.auto_server = None
        self.concurrency = 0
        self.inactive_servers = [Server(*args, **kwargs) 
            for _ in range(min(100, self.servers_per_worker))
        ]
        self.is_initialized = False
        self.active_check_interval = 60

    def _initialize(self):
        if self.is_initialized: return
        self.is_initialized = True
        self.num_sessions = 0
        asyncio.create_task(self._check_health())
        
    async def _check_health(self):
        await asyncio.sleep(60)
        merge(f"session/{self.name}", self.num_sessions,
            desc={"time_window_sum": "session start per minute"}
        )
        self.num_sessions = 0
        merge(f"server/{self.name}", len(self.servers),
            desc={"time_window_sum": "smoothed server num"}
        )
        asyncio.create_task(self._check_health())

    def _create_worker(self):
        if len(self.servers) >= self.servers_per_worker:
            raise Exception("Server exceeds max num.")
        return self.Server(*self.args, **self.kwargs)

    async def _check_active(self, session):
        await asyncio.sleep(self.active_check_interval)
        if not (server := self.servers.get(session, None)): 
            return
        interval = time.time() - server.__bray_atime
        if interval < self.active_check_interval:
            asyncio.create_task(self._check_active(session))
            return
        self.inactive_servers.append(server)
        self.servers.pop(session)
        print(f"Server with session={session} inactive.")

    async def start(self, session, data) -> bytes:
        server = self.servers.get(session, None)
        if server:
            raise Exception(f"Session {session} already started")
        try:
            server = self.inactive_servers.pop()
        except IndexError:
            server = self._create_worker()
        self.servers[session] = server
        try:
            server.__bray_atime = time.time()
            start_ret = await server.start(session, data)
        except:
            self.servers.pop(session, None)
            raise
        self.num_sessions += 1
        asyncio.create_task(self._check_active(session))
        return start_ret

    async def tick(self, session, data) -> bytes:
        server = self.servers.get(session, None)
        if not server:
            raise Exception(f"Session {session} not started.")
        server.__bray_atime = time.time()
        try:
            tick_ret = await server.tick(data)
        except:
            self.servers.pop(session, None)
            raise
        merge_time_ms(
            f"tick/{self.name}", server.__bray_atime)
        return tick_ret

    async def auto(self, data) -> bytes:
        tick_beg = time.time()
        while self.concurrency >= self.servers_per_worker:
            await asyncio.sleep(0.001)
        if self.auto_server is None:
            try:
                server = self.inactive_servers.pop()
            except IndexError:
                server = self._create_worker()
            self.auto_server = server
        try:
            self.concurrency += 1
            tick_ret = await self.auto_server.tick(data)
            self.concurrency -= 1
        except:
            self.auto_server = None
            raise
        merge_time_ms(f"tick/{self.name}", tick_beg)
        return tick_ret

    async def __call__(self, headers: Dict, body: bytes) -> bytes:
        if not self.is_initialized:
            self._initialize()
        step_kind = headers.get("step_kind", "auto")

        if step_kind == "auto":
            return await self.auto(body)

        session = headers.get("session")
        if session is None:
            raise Exception("session must be provided.")

        if step_kind == "tick":
            return await self.tick(session, body)

        if step_kind == "start":
            return await self.start(session, body)

        if step_kind != "stop":
            raise Exception("Unknown step_kind:", step_kind)

        server = self.servers.pop(session, None)
        if not server:
            raise Exception(f"Session {session} not started.")

        stop_ret = await server.stop(body)
        self.inactive_servers.append(server)
        return stop_ret


SERVER_GATEWAY: ServerGateway = None


def set_server_gateway(server_gateway: ServerGateway):
    global SERVER_GATEWAY
    SERVER_GATEWAY = server_gateway


async def handle_client(reader: StreamReader, writer: StreamWriter):
    async def handle(headers: Dict, body: bytes):
        global SERVER_GATEWAY
        try:
            data = await SERVER_GATEWAY(headers, body)
        except:
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()
            return
        if not isinstance(data, bytes):
            raise Exception("Server return must be bytes")
        session = headers["session"].encode()
        session_size, body_size = len(session), len(data)
        time = headers["time"]
        try:
            header = struct.pack("!3q", session_size, body_size, time)
            writer.write(header + session + data)
            await writer.drain()
        except:
            print(f"Fail to write Session {session}")
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()

    while not writer.is_closing():
        try:
            data = await reader.readexactly(8 * 6)
            (
                session_size,
                step_kind_size,
                key_size,
                token_size,
                body_size,
                time,
            ) = struct.unpack("!6q", data)
            data = await reader.readexactly(
                session_size + 
                step_kind_size + key_size + token_size + body_size
            )
        except (
            ConnectionResetError,
            asyncio.exceptions.IncompleteReadError,
        ):
            # print("Client disconnected")
            writer.close()
            await writer.wait_closed()
            return
        except:
            print("Fail to read from session")
            traceback.print_exc()
            writer.close()
            await writer.wait_closed()
            return
        session = data[0:session_size].decode()
        step_kind = data[session_size : session_size + step_kind_size]
        offset = session_size + step_kind_size
        key = data[offset : offset + key_size]
        offset += key_size
        token = data[offset : offset + token_size]
        body = data[offset + token_size :]
        headers = {
            "session": session,
            "step_kind": step_kind.decode(),
            "key": key.decode(),
            "token": token.decode(),
            "time": time,
        }
        asyncio.create_task(handle(headers, body))


@ray.remote(max_retries=-1)
def ServerWorker(name, port, Server, 
    args, kwargs, servers_per_worker, use_tcp, server, gateway
):
    server.register.remote(ray.util.get_node_ip_address())
    if gateway: gateway.register.remote("localhost", port)

    async def serve_tcp_gateway():
        server = await asyncio.start_server(
            handle_client, "0.0.0.0", port, reuse_port=True
        )
        async with server:
            await server.serve_forever()

    try:
        server_gateway = ServerGateway(
            name, Server, args, kwargs, servers_per_worker)
        set_server_gateway(server_gateway)
    except:
        print(f"Fail to start ServerGateway {name}")
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
            data = await server_gateway(headers, body)
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
    uvicorn.run(app, fd=sock.fileno(), 
        timeout_keep_alive=60 * 5, log_level=WARN)

    
@ray.remote(num_cpus=0)
class Server:
    def __init__(self, name, 
        Server: Type[Server], server_args, server_kwargs, port
    ):
        assert Server and port, f"Fail to start {name}"
        self.name, self.worker_ips = name, []
        self.Server = Server
        self.server_args, self.server_kwargs = server_args, server_kwargs
        self.port = port

    async def register(self, worker_ip):
        if worker_ip in self.worker_ips: return
        self.worker_ips.append(worker_ip)
        
    async def get_worker_ip(self, ip) -> str:
        if ip in self.worker_ips: return "localhost"
        interval = 1
        while not self.worker_ips:
            print(f"Wait {self.name} worker to be initialized")
            await asyncio.sleep(interval)
            interval = min(30, interval * 2)
        return random.sample(self.worker_ips, 1)[0]

    async def get_initialize_info(self, ip):
        return (self.Server, self.server_args, self.server_kwargs, 
            self.port, await self.get_worker_ip(ip))

SERVER = Server


class RemoteServer:
    """
    RemoteServer封装了一个Server服务，它会在Ray集群中创建多个Server实例，
    支持在集群内部和外部调用，支持TCP/Http协议和自动负载均衡
    """

    def __init__(
        self,
        name: str = "default",
        Server: Type[Server] = None, 
        server_args: Tuple = (), 
        server_kwargs: Dict = {},
        port: int = 8000,
        workers_per_node: int = 2,
        # num_workers: int = 2,
        cpus_per_worker: float = 1.0,
        gpus_per_worker: float = 0.0,
        memory_per_worker: int = 512,
        servers_per_worker: int = 10,
        use_tcp: bool = False,
        use_gateway: ["node", "head", None] = "node",
    ):
        """
        创建、初始化并启动一个RemoteServer，支持在集群内外调用
        Args:
            name:
        Server的名字，用于在Ray集群中标识Server
            Server:
        用户实现的Server类，需继承Server基类，实现tick接口
            server_args:
        Server类构造函数的位置参数，类型为Tuple
            server_kwargs:
        Server类构造函数的关键字参数，类型为Dict
            port: 
        RemoteServer 暴露给 Client 的服务端口
            workers_per_node:
        每个节点的 worker 数量，总 num_workers = num_nodes * workers_per_node
            num_workers: 
        RemoteServer 的 worker 数量
            cpus_per_worker: 
        每个 worker 的 CPU 占用量
            memory_per_worker: 
        每个 worker 的内存占用量，单位 MB
            servers_per_worker: 
        每个 worker 的 Server 数量，总 Servers = num_workers * servers_per_worker
            use_tcp: 
        是否使用 TCP 作为通信协议
            gateway: 
        ServerGateway 的位置，可以是 "node" 或 "head" 或 None
        """
        self.name, self.port = name, port
        self.Server = Server
        self.server_args, self.server_kwargs = server_args, server_kwargs
        self.workers_per_node = workers_per_node
        # self.num_workers = num_workers
        self.cpus_per_worker = cpus_per_worker
        self.gpus_per_worker = gpus_per_worker
        self.memory_per_worker = memory_per_worker
        self.servers_per_worker = servers_per_worker
        self.use_tcp, self.use_gateway = use_tcp, use_gateway

        self.server = SERVER.options(
            name=self.name, 
            get_if_exists=True, 
            scheduling_strategy=ray_scheduling_local(),
            max_concurrency=100000,
        ).remote(self.name, self.Server, 
            self.server_args, self.server_kwargs, self.port)

        if Server is not None:
            self.initialize_gateway_and_worker()

        (self.Server, self.server_args, self.server_kwargs, 
        self.port, self.worker_ip
        ) = ray.get(self.server.get_initialize_info.remote(
            ray.util.get_node_ip_address()))

        self.auto_server = self.Server(
            *self.server_args, **self.server_kwargs)
        self.sess = None
        self.url = f"http://{self.worker_ip}:{self.port}/step"
        self.auto_server.tick = self.tick

    def initialize_gateway_and_worker(self):
        print(f"Starting Server {self.name}...")

        self.node_ids = [
            node["NodeID"] for node in ray.nodes() if node["Alive"]]
        gateway = (
        None if self.use_gateway != "head"
        else ray.remote(Gateway).options(
            num_cpus=0,
            scheduling_strategy=ray_scheduling_local()
        ).remote(self.port)
        )
        self.gateways = [
        gateway if gateway or not self.use_gateway
        else ray.remote(Gateway).options(
            num_cpus=0,
            scheduling_strategy=ray_scheduling_local(node_id)
        ).remote(self.port)
        for node_id in self.node_ids
        ]
        self.workers = [
        ServerWorker.options(
            num_cpus=self.cpus_per_worker,
            num_gpus=self.gpus_per_worker,
            memory=self.memory_per_worker * 1024 * 1024,
            scheduling_strategy=ray_scheduling_local(node_id),
        ).remote(
            self.name,
            self.port + (i + 1 if self.use_gateway else 0),
            self.Server,
            self.server_args,
            self.server_kwargs,
            self.servers_per_worker,
            self.use_tcp,
            self.server,
            gateway,
        )
        for node_id, gateway in zip(self.node_ids, self.gateways)
        for i in range(self.workers_per_node)
        ]
        ray.get([gateway.serve.remote() 
            for gateway in self.gateways if gateway])
        print(f"Server {self.name} started at port {self.port}.")

    def tick(self, data: bytes) -> bytes:
        if not self.sess:
            import requests
            self.sess = requests.Session()
        tick_beg = time.time()
        res = self.sess.post(
            url=self.url,
            # headers={
            #     "session": self.session,
            #     "step_kind": kind,
            # },
            data=data,
            # timeout=self.timeout,
        )
        merge_time_ms(
            f"tick/{self.name}", tick_beg, mode="remote")
        if res.status_code != 200:
            raise Exception(res.text)
        return res.content

    def step(self, *args, **kwargs):
        step_beg = time.time()
        outputs = self.auto_server.step(*args, **kwargs)
        merge_time_ms(f"step/{self.name}", step_beg)
        return outputs
