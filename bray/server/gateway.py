from asyncio import StreamReader, StreamWriter
import asyncio

class Gateway:
    def __init__(self, port: int):
        self.server_addrs, self.server_index = [], 0
        self.port = port

    async def start(self):
        gateway = await asyncio.start_server(
            self.handle, "0.0.0.0", self.port, reuse_port=True
        )
        async with gateway:
            await gateway.serve_forever()

    async def serve(self):
        asyncio.create_task(self.start())
        await asyncio.sleep(3)

    async def register(self, ip: str, port: int):
        if (ip, port) in self.server_addrs:
            return
        self.server_addrs.append((ip, port))

    async def _handle(
        self, reader: StreamReader, writer: StreamWriter):
        try:
            while data := await reader.read(4096):
                writer.write(data)
                await writer.drain()
        except (
            ConnectionResetError,
            asyncio.exceptions.IncompleteReadError,
        ):
            pass
            # print("Client disconnected")
        except:
            import traceback
            traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()

    async def handle(
        self, reader: StreamReader, writer: StreamWriter):
        if not self.server_addrs:
            print("No Gateway server available")
            writer.close()
            return await writer.wait_closed()

        index = self.server_index % len(self.server_addrs)
        self.server_index += 1
        ip, port = self.server_addrs[index]

        try:
            r, w = await asyncio.open_connection(ip, port)
        except:
            writer.close()
            return await writer.wait_closed()

        asyncio.create_task(self._handle(reader, w))
        asyncio.create_task(self._handle(r, writer))
