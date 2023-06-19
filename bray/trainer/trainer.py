import ray

from horovod.ray import RayExecutor

from bray.buffer.buffer import Buffer, RemoteBuffer


def buffer_generator(buffer):
    while True:
        yield ray.get(buffer.pop.remote())


class TrainerWorker:
    def __init__(self, Trainer, config):
        self.trainer = Trainer(config)
        self.config = config
        self.buffers = {}

    def train(self, remote_model, buffer_name):
        model = remote_model.get_model()
        replays = buffer_generator(self.buffers[buffer_name])
        self.trainer.train(model, replays)

    def new_buffer(self, buffer_name):
        scheduling_local = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
        buffer = Buffer.options(
            scheduling_strategy=scheduling_local
        ).remote()
        self.buffers[buffer_name] = buffer
        return buffer


class RemoteTrainer:
    def __init__(self, Trainer, config):
        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(settings, num_workers=2, use_gpu=False)
        self.executor.start(executable_cls=TrainerWorker, executable_args=[Trainer, config])
        self.remote_buffers = {}

    def train(self, remote_model, remote_buffer):
        buffer_name = remote_buffer.get_name()
        return self.executor.execute(lambda worker: worker.train(remote_model, buffer_name))

    def new_buffer(self, name):
        buffers = self.executor.execute(lambda worker: worker.new_buffer(name))
        remote_buffer = RemoteBuffer(name, buffers)
        self.remote_buffers[name] = remote_buffer
        return remote_buffer
