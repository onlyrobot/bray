import logging

from .model import AtariModel
from .actor import AtariActor
from .trainer import AtariTrainer
import bray

model = AtariModel()

remote_model = bray.RemoteModel(model=model)

remote_trainer = bray.RemoteTrainer(Trainer=AtariTrainer, config=None)

remote_buffer = remote_trainer.new_buffer(name="buffer1")

agent = bray.Agent(remote_model=remote_model, remote_buffer=remote_buffer)

remote_actor = bray.RemoteActor(
    port=8000, Actor=AtariActor, agents={"agent1": agent}, config=None
)

remote_trainer.train(remote_model=remote_model, remote_buffer=remote_buffer)
