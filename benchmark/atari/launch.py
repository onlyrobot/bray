import numpy as np
import bray

from .model import AtariModel
from .actor import AtariActor
from .trainer import train_atari

bray.init(project="./atari-pengyao", trial="ppo-v0")

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=(np.random.randn(42, 42, 4).astype(np.float32),),
)

remote_buffer = bray.RemoteBuffer("atari_buffer")

remote_trainer = bray.RemoteTrainer(
    use_gpu=None,
    num_workers=None,
)

remote_trainer.train(
    train=train_atari,
    model="atari_model",
    buffer="atari_buffer",
    batch_size=8,
    weights_publish_interval=1,
    num_steps=1000000,
)

remote_actor = bray.RemoteActor(port=8000)

remote_actor.serve(
    Actor=AtariActor,
    model="atari_model",
    buffer="atari_buffer",
)

bray.run_until_asked_to_stop()
