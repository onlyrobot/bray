import asyncio
from asyncio import StreamReader, StreamWriter

import ray


@ray.remote
class Gateway:
    def __init__(self, port: int):
        self.port = port
        self.server_addrs, self.server_index = [], 0
        asyncio.create_task(self.start())

    async def start(self):
        gateway = await asyncio.start_server(
            self.handle, "0.0.0.0", self.port, reuse_port=True
        )
        async with gateway:
            await gateway.serve_forever()

    async def register(self, ip: str, port: int):
        self.server_addrs.append((ip, port))

    async def handle(self, reader: StreamReader, writer: StreamWriter):
        if not self.server_addrs:
            writer.close()
            return await writer.wait_closed()

        index = self.server_index % len(self.server_addrs)
        self.server_index += 1
        ip, port = self.server_addrs[index]
        is_ok = True

        async def handle(r: StreamReader, w: StreamWriter):
            try:
                nonlocal is_ok
                while is_ok:
                    w.write(await r.read(4096))
                    await w.drain()
            except:
                w.close()
                await w.wait_closed()
                is_ok = False

        try:
            r, w = await asyncio.open_connection(ip, port)
        except:
            writer.close()
            return await writer.wait_closed()

        asyncio.create_task(handle(reader, w))
        asyncio.create_task(handle(r, writer))
