import numpy as np
import bray
import torch

from .model import AtariModel
from .actor import AtariActor
from .trainer import AtariTrainer

bray.init(project="./atari-pengyao", trial="ppo-v0")

model_inputs = {"image": torch.rand(1, 42, 42, 4, dtype=torch.float32)}

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=(model_inputs,),
    num_workers=0,
    local_mode=True,
    use_onnx="train",
)
bray.add_graph(remote_model.get_model().eval(), model_inputs)
bray.set_tensorboard_step(remote_model)

remote_buffer = bray.RemoteBuffer(
    "atari_buffer",
    size=128,
    batch_size=8,
    num_workers=2,
)

remote_trainer = bray.RemoteTrainer(
    name="kitty_trainer", 
    Trainer=AtariTrainer,
    config=None,
    use_gpu=None,
    num_workers=1,
    remote_model=remote_model,
    remote_buffer=remote_buffer,
    batch_size=8,
    weights_publish_interval=1,
    num_steps=1000000,
)

remote_actor = bray.RemoteActor(port=8000, num_workers=4)

remote_actor.serve(
    Actor=AtariActor,
    remote_model=remote_model,
    remote_buffer=remote_buffer,
)

bray.run_until_asked_to_stop()
