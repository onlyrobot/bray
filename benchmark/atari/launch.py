import bray

from .model import AtariModel
from .actor import AtariActor
from .trainer import train_atari

bray.init(project="./atari-pengyao", trial="ppo-v0")

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    override=True,
)

remote_actor = bray.RemoteActor(port=8000)

remote_actor.serve(
    Actor=AtariActor,
    model="atari_model",
    buffer="atari_buffer",
)

remote_trainer = bray.RemoteTrainer(
    num_workers=4,
    use_gpu=False,
)

remote_trainer.train(
    train=train_atari,
    model="atari_model",
    buffer="atari_buffer",
    weights_publish_interval=4,
    num_steps=100000,
)

bray.run_until_asked_to_stop()