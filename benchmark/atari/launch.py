import numpy as np
import bray

from .model import AtariModel
from .actor import AtariActor
from .trainer import train_atari

bray.init(project="./atari-pengyao", trial="ppo-v0")

model_inputs = {"image": np.random.randn(42, 42, 4).astype(np.float32)}

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=(model_inputs,),
    num_workers=2,
    local_mode=False,
    use_onnx="train",
)
bray.add_graph(remote_model.get_model(), remote_model.get_torch_forward_args())
bray.set_tensorboard_step(remote_model.name)

remote_buffer = bray.RemoteBuffer(
    "atari_buffer",
    size=128,
    batch_size=8,
    num_workers=2,
)

remote_trainer = bray.RemoteTrainer(
    use_gpu=None,
    num_workers=None,
)

remote_trainer.train(
    train=train_atari,
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
