from .model import AtariModel
from .actor import AtariActor
from .trainer import AtariTrainer
import bray

model = AtariModel()

remote_model = bray.RemoteModel(model)

remote_trainer = bray.RemoteTrainer(AtariTrainer, None)

remote_buffer = remote_trainer.new_buffer("buffer1")

agent = bray.Agent(remote_model, remote_buffer)

remote_actor = bray.RemoteActor(AtariActor, {"agent1": agent}, None)
remote_actor.serve_background()

remote_trainer.train(remote_model, remote_buffer)