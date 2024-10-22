import bray


class Actor1(bray.Actor):
    def __init__(self, name, config: dict):
        raise NotImplementedError

    async def start(self, session, data: bytes) -> bytes:
        raise NotImplementedError

    async def tick(self, data: bytes) -> bytes:
        raise NotImplementedError

    async def stop(self, data: bytes) -> bytes:
        raise NotImplementedError
