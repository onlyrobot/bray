import bray


class Actor1(bray.Actor):
    def __init__(self):
        raise NotImplementedError

    async def start(self, game_id, data: bytes) -> bytes:
        raise NotImplementedError

    async def tick(self, data: bytes) -> bytes:
        raise NotImplementedError

    async def stop(self, data: bytes) -> bytes:
        raise NotImplementedError
