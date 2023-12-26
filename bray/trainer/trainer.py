import torch
from typing import Type
import math
import time
import ray
import numpy as np
from bray.trainer.base import Trainer
from bray.model.model import RemoteModel, get_torch_model_weights
from bray.buffer.buffer import RemoteBuffer
from bray.buffer.utils import (
    BatchBuffer,
    TorchTensorBuffer,
    PrefetchBuffer,
    CallbackBuffer,
    SampleBuffer,
)
from bray.metric.metric import merge_time_ms


def train(
    name: str,
    Trainer: Type[Trainer],
    remote_model: RemoteModel,
    remote_buffers: dict[str:RemoteBuffer],
    buffer_weights: [float] = None,
    batch_size: int = 1,
    batch_kind: ["concate", "stack", None] = "concate",
    prefetch_size: int = 1,
    max_reuse: int = 0,
    clip_grad_max_norm: float = 1.0,
    weights_publish_interval: int = 1,
    num_steps: int = 100000000,
    remote_eval_buffer: RemoteBuffer = None,
    eval_interval: int = 1000,
    eval_steps: int = 10,
):
    # initialize horovod and torch
    import horovod.torch as hvd

    hvd.init()
    np.random.seed(0)
    device = torch.device(
        hvd.local_rank() if torch.cuda.is_available() else "cpu",
    )
    # initialize model and and trainer
    model = remote_model.get_model()
    model.to(device=device)
    trainer = Trainer(model)
    # initialize optimizer
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        model.named_parameters(),
    )
    # initialize buffer
    # total batch size = buffer batch size * batch_size * horovod size
    buffers = list(remote_buffers.values())
    if remote_eval_buffer:
        buffers.append(remote_eval_buffer)
    names = [buffer.name for buffer in buffers]
    buffers = [
        BatchBuffer(
            b,
            batch_size=batch_size,
            kind=batch_kind,
        )
        for b in buffers
    ]
    buffers = [
        TorchTensorBuffer(
            b,
            device=device,
        )
        for b in buffers
    ]
    buffers = [
        CallbackBuffer(
            b,
            callback=trainer.replay_handler,
        )
        for b in buffers
    ]
    buffers = [
        PrefetchBuffer(
            buffers[i],
            size=prefetch_size,
            max_reuse=max_reuse,
            name=names[i],
        )
        if prefetch_size > 0 or max_reuse > 0
        else buffers[i]
        for i in range(len(buffers))
    ]
    if remote_eval_buffer:
        eval_buffer = buffers.pop()
    if len(buffers) > 1:
        buffer = SampleBuffer(buffers, buffer_weights)
    else:
        buffer = buffers[0]

    def eval_one_step():
        beg = time.time()
        replay = next(eval_buffer)
        merge_time_ms(f"replay/{name}", beg)
        beg = time.time()
        trainer.eval(replay)
        merge_time_ms(f"eval/{name}", beg)

    def eval_at_step(step):
        print(f"Trainer {name} eval at step {step}")
        model.eval()
        for _ in range(eval_steps):
            eval_one_step()
        model.train()
        print(f"Trainer {name} eval done")

    restore_step = remote_model.step
    print("Trainer {} start from step {}".format(name, restore_step))
    start_step = restore_step + 1
    for i in range(start_step, start_step + num_steps):
        if remote_eval_buffer and i % eval_interval == 0:
            eval_at_step(i)
        beg = time.time()
        replay = next(buffer)
        merge_time_ms(f"replay/{name}", beg)
        beg = time.time()
        optimizer.zero_grad()
        loss = trainer.loss(replay)
        loss.backward()
        optimizer.synchronize()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_grad_max_norm,
        )
        with optimizer.skip_synchronize():
            optimizer.step()
        if hvd.rank() != 0:
            continue
        merge_time_ms(f"train/{name}", beg)
        if i % 10 == 0:
            print(f"Trainer {name} train step {i}, loss: {loss.item()}")
        if i % weights_publish_interval != 0:
            continue
        remote_model.publish_weights(
            get_torch_model_weights(model),
        )
    print(f"Trainer {name} train all {num_steps} steps done!")


class RemoteTrainer:
    """
    这个类用于在多个节点上训练模型，它会在多个节点上创建 Trainer 的实例，
    然后调用 train 函数
    """

    def __init__(
        self,
        name: str = "default",
        use_gpu: bool = None,
        num_workers: int = None,
        cpus_per_worker: int = None,
        total_cpus_ratio: float = 0.75,
        framework: ["torch", "tensorflow"] = "torch",
    ):
        """
        Args:
            use_gpu: 是否使用GPU，如果不指定则会自动判断
            num_workers: 训练的 worker 数量，如果不指定则会自动计算
            cpus_per_worker: 每个节点的CPU核心数，仅当 use_gpu 为 False 时有效
            total_cpus_ratio: 训练节点的总CPU核心数占用比例，仅当 use_gpu 为 False 时有效
        """
        self.name = name
        from horovod.ray import RayExecutor

        total_cpus = ray.available_resources()["CPU"]
        total_gpus = ray.available_resources().get("GPU", 0)

        use_gpu = use_gpu if use_gpu == True else use_gpu is None and total_gpus > 0

        if not use_gpu:
            trainer_cpus = max(1, int(total_cpus * total_cpus_ratio))
            if not num_workers:
                num_workers = int(math.sqrt(trainer_cpus))
            if not cpus_per_worker:
                cpus_per_worker = trainer_cpus // num_workers
        else:
            cpus_per_worker = 2
            num_workers = num_workers if num_workers else total_gpus

        print(
            f"Trainer {name} start with {num_workers} {'GPU' if use_gpu else 'CPU'} workers, "
            + f"{cpus_per_worker} cpus per worker"
        )

        settings = RayExecutor.create_settings()
        self.executor = RayExecutor(
            settings,
            num_workers=num_workers,
            cpus_per_worker=cpus_per_worker,
            use_gpu=use_gpu,
        )
        self.executor.start()

        def init_torch():
            import torch

            torch.set_num_interop_threads(cpus_per_worker)
            torch.set_num_threads(cpus_per_worker)

        def init_tensorflow():
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            threading = tf.config.threading
            threading.set_inter_op_parallelism_threads(cpus_per_worker)
            threading.set_intra_op_parallelism_threads(cpus_per_worker)

        init_framework = init_torch if framework == "torch" else init_tensorflow
        ray.get(self.executor.run_remote(init_framework))

    def train(self, train: callable, *args, **kwargs) -> list:
        """
        在多个节点上执行训练函数
        Args:
            train: 训练函数
            *args: 训练函数的位置参数
            **kwargs: 训练函数的关键字参数
        Returns:
            训练函数在每个节点上的返回值，可以通过 ray.get() 获取
        """
        return self.executor.run_remote(train, args, kwargs)
