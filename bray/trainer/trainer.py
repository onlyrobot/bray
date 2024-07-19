import torch
from typing import Type, List, Dict
import time, os
import ray
import numpy as np
from bray.trainer.base import Trainer
from bray.model.model import RemoteModel, get_torch_model_weights
from bray.buffer.buffer import RemoteBuffer
from bray.buffer.utils import (
    ListBuffer,
    StackBuffer,
    ConcateBuffer,
    TorchTensorBuffer,
    PrefetchBuffer,
    CallbackBuffer,
    SampleBuffer,
)
from bray.master.master import merge_time_ms
from bray.utils import ray_scheduling_local
from ray.train.torch import TorchTrainer
from ray.train import (
    RunConfig, ScalingConfig, get_context, CheckpointConfig
)

# from ray.air.config import (
#     RunConfig, ScalingConfig, CheckpointConfig
# )
# from ray.air import session

# get_context = lambda: session

def train_loop_per_worker(
    name: str,
    Trainer: Type[Trainer],
    config: Dict,
    remote_model: RemoteModel,
    remote_models: List[RemoteModel] = None,
    remote_buffer: RemoteBuffer = None,
    remote_buffers: Dict[str, RemoteBuffer] = None,
    buffer_weights: List[float] = None,
    batch_size: int = 1,
    batch_kind: ["concate", "stack", "list", None] = "concate",
    prefetch_size: int = 1,
    max_reuse: int = 0,
    learning_rate: float = None,
    clip_grad_max_norm: float = 1.0,
    weights_publish_interval: int = 1,
    num_steps: int = 100000000,
    remote_eval_buffer: RemoteBuffer = None,
    eval_interval: int = 1000,
    eval_steps: int = 10,
    optimizer_step_interval: int = 1,
):
    # initialize torch trainer worker and train loop
    world_size = get_context().get_world_size()
    world_rank = get_context().get_world_rank()
    np.random.seed(0)
    device = ray.train.torch.get_device()
    print(f"Trainer {name} worker {world_rank} starting at device {device}")
    # initialize model and and trainer
    model = remote_model.get_model()
    # Instantiate and prepare model for training.
    model = ray.train.torch.prepare_model(model)

    # initialize optimizer
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, 
        lr=learning_rate or 5e-4)
    trainer = Trainer(name, config, model, optimizer)
    # initialize buffer
    # total batch size = buffer batch size * batch_size * horovod size
    if remote_buffer: buffers = [remote_buffer]
    else:
        assert remote_buffers, "remote_buffers is None"
        buffers = list(remote_buffers.values())
    if remote_eval_buffer:
        buffers.append(remote_eval_buffer)
    buffer_batch_size = buffers[0].batch_size or 1
    names = [buffer.name for buffer in buffers]
    
    BatchBuffer = lambda b, size: b
    if batch_kind == "stack": BatchBuffer = StackBuffer
    if batch_kind == "list": BatchBuffer = ListBuffer
    if batch_kind == "concate":
        BatchBuffer = ConcateBuffer
    buffers = [BatchBuffer(b, size=batch_size) for b in buffers]
    buffers = [CallbackBuffer(b, callback=trainer.handle) 
        for b in buffers]
    buffers = [TorchTensorBuffer(b, device=device) 
        for b in buffers]
    buffers = [PrefetchBuffer(buffers[i], 
            max(1, prefetch_size), max_reuse, names[i])
        if prefetch_size > 0 or max_reuse > 0
        else buffers[i]
        for i in range(len(buffers))]
    if remote_eval_buffer: eval_buffer = buffers.pop()
    if remote_buffer: buffer = buffers.pop()
    elif len(buffers) > 1:
        buffer = SampleBuffer(buffers, buffer_weights)
    else: buffer = buffers[0]

    def eval_with_metric():
        next_beg, replay = time.time(), next(eval_buffer)
        merge_time_ms(f"eval/replay/{name}", next_beg)
        eval_beg = time.time()
        trainer.eval(replay)
        merge_time_ms(f"eval/{name}", eval_beg)

    def eval_at_step(step):
        print(f"Trainer {name} eval at step {step}")
        model.eval()
        with torch.no_grad():
            for _ in range(eval_steps): eval_with_metric()
        model.train()
        print(f"Trainer {name} eval done")

    model_step = remote_model.step
    if world_rank == 0:
        print(f"Trainer {name} start with {remote_model.name}" + 
            f"at step {model_step}")
    optimizer.zero_grad()
    train_step_beg = time.time()
    for i in range(1, 1 + num_steps):
        if remote_eval_buffer and i % eval_interval == 0:
            eval_at_step(i)
        next_beg, replay = time.time(), next(buffer)
        if world_rank == 0:
            merge_time_ms(f"replay/{name}", next_beg)
        model_step += world_size * batch_size * buffer_batch_size
        # zero grad, loss backward and optimizer step
        loss = trainer.loss(replay, model_step)
        if optimizer_step_interval > 1:
            loss /= optimizer_step_interval
        loss.backward()
        if i % optimizer_step_interval != 0: continue
        # TODO(pengyao): is clip in ddp ok?
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step(), optimizer.zero_grad()
        if world_rank != 0: continue
        merge_time_ms(f"train/{name}", train_step_beg)
        train_step_beg = time.time()
        optimizer_step = i // optimizer_step_interval
        if optimizer_step % 10 == 0:
            print(f"Trainer {name} step {i}, loss: {loss.item()}")
        if optimizer_step % weights_publish_interval != 0:
            continue
        publish_beg = time.time()
        module = model.module if world_size > 1 else model
        remote_model.publish_weights(
            get_torch_model_weights(module), model_step
        )
        merge_time_ms(f"publish/{name}", publish_beg)
        train_step_beg = time.time()
    print(f"Trainer {name} train all {num_steps} steps done!")


def train(name: str, torch_trainer: TorchTrainer):
    @ray.remote(num_cpus=0)
    def Trainer():
        try: return torch_trainer.fit()
        except Exception as e: pass
        print(f"Fail to fit Trainer {name}: {e}")
        raise e
    return Trainer.options(
        scheduling_strategy=ray_scheduling_local()).remote()


def RemoteTrainer(
    name: str , Trainer: Type[Trainer], use_gpu: bool = None, 
    num_workers: int = None, **kwargs
):
    """
    Args:
        Trainer: 被封装的Trainer类，该类继承自 bray.Trainer
        use_gpu: 是否使用GPU，如果不指定则会自动判断
        num_workers: 训练的 worker 数量，如果不指定则会自动计算
        **kwargs: 训练函数train_loop_per_worker的关键字参数
    """
    total_gpus = ray.available_resources().get("GPU", 0)

    if use_gpu is None: use_gpu = total_gpus > 0

    num_workers = num_workers if num_workers else total_gpus

    def train_wrapper(kwargs):
        train_loop_per_worker(name=name, Trainer=Trainer, **kwargs)

    scaling_config = ScalingConfig(
        num_workers=num_workers, use_gpu=use_gpu,
        trainer_resources={"CPU": 0}, resources_per_worker={"CPU": 0}
    )

    trial_path = ray.get_runtime_context().namespace
    storage_path = os.path.join(trial_path, name)
    checkpoint_config = CheckpointConfig(
        num_to_keep=1, checkpoint_frequency=0)
    run_config = RunConfig(
        # storage_path=storage_path, checkpoint_config=checkpoint_config
    )

    torch_trainer = TorchTrainer(
        train_wrapper, scaling_config=scaling_config, 
        train_loop_config=kwargs, run_config=run_config,
    )
    print(f"Trainer {name} starting with {num_workers} " + 
            f"{'GPU' if use_gpu else 'CPU'} workers, ")
    return train(name, torch_trainer)