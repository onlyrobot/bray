from typing import Iterator

import ray

from horovod.ray import RayExecutor

from bray.buffer.buffer import Buffer, RemoteBuffer
from bray.trainer.base import Trainer
from bray.model.model import RemoteModel, NestedArray


def buffer_generator(buffer: Buffer) -> Iterator[NestedArray]:
    while True:
        yield ray.get(buffer.pop.remote())


class TrainerWorker:
    def __init__(self, Trainer: type[Trainer], config):
        self.trainer = Trainer(config)
        self.buffers = {}

    def train(self, remote_model, buffer_name):
        replays = buffer_generator(self.buffers[buffer_name])
        self.trainer.train(remote_model, replays)

    def new_buffer(self, buffer_name):
        scheduling_local = (
            ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            )
        )
        buffer = Buffer.options(scheduling_strategy=scheduling_local).remote()
        self.buffers[buffer_name] = buffer
        return buffer


class RemoteTrainer:
    """
    分布式训练器，用于在多个节点上训练模型，封装好了 RayExecutor 的使用
    """

    def __init__(self, Trainer: type[Trainer], config: any):
        """
        Args:
            Trainer: bray.Trainer 的子类型，用于训练模型
            config: Trainer 的配置，传给 Trainer 的构造函数
        """
        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(settings, num_workers=2, use_gpu=False)
        self.executor.start(
            executable_cls=TrainerWorker, executable_args=[Trainer, config]
        )
        self.remote_buffers = {}

    def train(self, remote_model: RemoteModel, remote_buffer: RemoteBuffer):
        """
        在指定的RemoteBuffer上训练模型
        """
        buffer_name = remote_buffer.get_name()
        return self.executor.execute(
            lambda worker: worker.train(remote_model, buffer_name)
        )

    def new_buffer(self, name: str) -> RemoteBuffer:
        """
        创建一个新的RemoteBuffer，该RemoteBuffer可以从任意节点push数据，
        然后被当前RemoteTrainer用于训练
        """
        buffers = self.executor.execute(lambda worker: worker.new_buffer(name))
        remote_buffer = RemoteBuffer(name, buffers)
        self.remote_buffers[name] = remote_buffer
        return remote_buffer
