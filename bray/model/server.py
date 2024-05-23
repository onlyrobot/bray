import time

from bray.utils.nested_array import NestedArray
from bray.server.base import Server
from bray.model.model import RemoteModel
from bray.master.master import merge_time_ms

# from ray import cloudpickle
import pickle as cloudpickle

class ModelServer(Server):
    # async def tick(self, data: bytes) -> bytes:
    #     load_beg = time.time()
    #     name, args, batch, kwargs = cloudpickle.loads(data)
    #     merge_time_ms("pickle/load", load_beg)
    #     outputs = await RemoteModel(name).forward(
    #         *args, batch=batch, **kwargs)
    #     return cloudpickle.dumps(outputs)

    # def step(self, name, *args, batch=True, **kwargs):
    #     """参数和返回值同RemoteModel的forward方法"""
    #     inputs = cloudpickle.dumps((name, args, batch, kwargs))
    #     outputs = self.tick(inputs)
    #     return cloudpickle.loads(outputs)

    async def tick(self, data: bytes) -> bytes:
        load_beg = time.time()
        name, args, batch, kwargs = cloudpickle.loads(data)
        merge_time_ms("pickle/load", load_beg)
        outputs = await RemoteModel(name).forward(
            *args, batch=batch, **kwargs)
        return cloudpickle.dumps(outputs)

    def step(self, name, *args, batch=True, **kwargs):
        """参数和返回值同RemoteModel的forward方法"""
        inputs = cloudpickle.dumps((name, args, batch, kwargs))
        outputs = self.tick(inputs)
        return cloudpickle.loads(outputs)