from horovod.ray import RayExecutor

from bray.buffer.buffer import RemoteBuffer, BufferIterator
from bray.trainer.base import Trainer
from bray.model.model import RemoteModel


class TrainerWorker:
    def __init__(self, Trainer: type[Trainer], config):
        self.trainer = Trainer(config)

    def train(self, remote_model, remote_buffer, num_steps):
        buffer_worker = remote_buffer.new_local_worker()
        buffer = BufferIterator(buffer_worker)
        self.trainer.train(remote_model, buffer, num_steps)


class RemoteTrainer:
    """
    这个类用于在多个节点上训练模型，它会在多个节点上创建 Trainer 的实例，
    然后调用 Trainer.train 函数
    """

    def __init__(
        self, num_workers: int, use_gpu: bool, Trainer: type[Trainer], config=None
    ):
        """
        Args:
            Trainer: bray.Trainer 的子类型，用于训练模型
            config: Trainer 的配置，传给 Trainer 的构造函数
        """
        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(settings, num_workers=num_workers, use_gpu=use_gpu)
        self.executor.start(
            executable_cls=TrainerWorker, executable_args=[Trainer, config]
        )

    def train(
        self,
        remote_model: RemoteModel,
        remote_buffer: RemoteBuffer,
        num_steps: int = -1,
    ):
        """
        在指定的RemoteBuffer上训练模型
        """
        num_steps = 1000000 if num_steps == -1 else num_steps
        return self.executor.execute(
            lambda worker: worker.train(remote_model, remote_buffer, num_steps)
        )
