import numpy as np
import bray

from .model import AtariModel
from ..actor import AtariActor
from .trainer import train_atari

bray.init(project="./atari-pengyao", trial="tf-ppo-v0", num_gpus=1)

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel,
    forward_args=({"image": np.random.randn(42, 42, 4).astype(np.float32)},),
    num_workers=0,
    local_mode=True,
    # gpus_per_worker=1,
    # use_onnx="train",
)

# bray.set_tensorboard_step(remote_model.name)

remote_buffer = bray.RemoteBuffer("atari_buffer")

remote_trainer = bray.RemoteTrainer(
    use_gpu=True,
    num_workers=1,
    framework="tensorflow",
)

remote_trainer.train(
    train=train_atari,
    remote_model=remote_model,
    remote_buffer=remote_buffer,
    batch_size=32,
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
