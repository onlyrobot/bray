import ray
import asyncio

@ray.remote
class Buffer:
    def __init__(self):
        self.replays = []

    async def push(self, replay):
        print("push")
        self.replays.append(replay)

    async def pop(self):
        print("pop")
        while len(self.replays) == 0:
            await asyncio.sleep(0.5)
        print("poped")
        return self.replays.pop()


class RemoteBuffer:
    def __init__(self, name, buffers):
        self.name = name
        self.buffers = buffers
        self.push_index = 0

    def push(self, replay):
        push_index = self.push_index % len(self.buffers)
        self.push_index += 1
        return ray.get(self.buffers[push_index].push.remote(replay))
    
    def get_name(self):
        return self.name