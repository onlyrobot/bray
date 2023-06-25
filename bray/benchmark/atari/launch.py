from .model import AtariModel
from .actor import AtariActor
from .trainer import AtariTrainer
import bray

remote_model = bray.RemoteModel(name="model1", model=AtariModel())

remote_buffer = bray.RemoteBuffer(name="buffer1")

agent = bray.Agent(remote_model=remote_model, remote_buffer=remote_buffer)

remote_actor = bray.RemoteActor(Actor=AtariActor, agents={"agent1": agent}, config=None)

remote_actor.serve(port=8000, background=True)

remote_trainer = bray.RemoteTrainer(
    num_workers=4, use_gpu=False, Trainer=AtariTrainer, config=None
)

remote_trainer.train(
    remote_model=remote_model, remote_buffer=remote_buffer, num_steps=1000000
)
