import numpy as np
import bray

from .model import AtariModel
from .actor import AtariActor

bray.init(project="./atari-pengyao", trial="ppo-v0")

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=(np.random.randn(42, 42, 4).astype(np.float32),),
    use_onnx=True,
)

remote_actor = bray.RemoteActor(port=8000)

remote_actor.serve(
    Actor=AtariActor,
    model="atari_model",
    buffer=None,
)

bray.run_until_asked_to_stop()
