import numpy as np
import bray

from .model import GridShootingModel
from .actor import GridShootingActor
from .trainer import train_grid_shooting

bray.init(project="./grid-shooting", trial="ppo-v0")

input_dict = {
    "obs": np.random.randn(56).astype(np.float32),
    "action_mask": np.ones(9).astype(np.float32),
}

remote_model = bray.RemoteModel(
    name="grid_shooting_model",
    model=GridShootingModel(),
    forward_args=(input_dict,),
    checkpoint_interval=1000,
    num_workers=0,
    local_mode=True,
    use_onnx="train",
)

remote_buffer = bray.RemoteBuffer("grid_shooting_buffer")

remote_trainer = bray.RemoteTrainer(
    use_gpu=None,
    num_workers=None,
)

remote_trainer.train(
    train=train_grid_shooting,
    remote_model=remote_model,
    remote_buffer=remote_buffer,
    batch_size=64,
    weights_publish_interval=1,
    num_steps=1000000,
)

remote_actor = bray.RemoteActor(port=8000, num_workers=4)

remote_actor.serve(
    Actor=GridShootingActor,
    remote_model=remote_model,
    remote_buffer=remote_buffer,
    target_step_interval=1000,
)

bray.run_until_asked_to_stop()
