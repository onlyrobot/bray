import asyncio
from asyncio import StreamReader, StreamWriter
import traceback

import ray


@ray.remote(num_cpus=0)
class Gateway:
    def __init__(self, port: int):
        self.port = port
        self.server_addrs, self.server_index = [], 0

    async def start(self):
        gateway = await asyncio.start_server(
            self.handle, "0.0.0.0", self.port, reuse_port=True
        )
        async with gateway:
            await gateway.serve_forever()

    async def serve(self):
        asyncio.create_task(self.start())

    async def register(self, ip: str, port: int):
        if (ip, port) in self.server_addrs:
            return
        self.server_addrs.append((ip, port))

    async def handle(self, reader: StreamReader, writer: StreamWriter):
        if not self.server_addrs:
            print("No Actor server available")
            writer.close()
            return await writer.wait_closed()

        index = self.server_index % len(self.server_addrs)
        self.server_index += 1
        ip, port = self.server_addrs[index]

        async def handle(r: StreamReader, w: StreamWriter):
            try:
                while data := await r.read(4096):
                    w.write(data)
                    await w.drain()
            except (
                ConnectionResetError,
                asyncio.exceptions.IncompleteReadError,
            ):
                pass
                # print("Client disconnected")
            except:
                traceback.print_exc()
            finally:
                w.close()
                await w.wait_closed()

        try:
            r, w = await asyncio.open_connection(ip, port)
        except:
            writer.close()
            return await writer.wait_closed()

        asyncio.create_task(handle(reader, w))
        asyncio.create_task(handle(r, writer))
