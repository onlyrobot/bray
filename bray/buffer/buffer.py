import ray
import asyncio

@ray.remote
class Buffer:
    def __init__(self):
        self.replays = []

    async def push(self, replay):
        self.replays.append(replay)

    async def pop(self):
        while len(self.replays) == 0:
            await asyncio.sleep(0.02)
        return self.replays.pop()


class RemoteBuffer:
    def __init__(self, name, buffers):
        self.name = name
        self.buffers = buffers
        self.push_index = 0

    def push(self, replay):
        push_index = self.push_index % len(self.buffers)
        self.push_index += 1
        self.buffers[push_index].push.remote(replay)
    
    def get_name(self):
        return self.name