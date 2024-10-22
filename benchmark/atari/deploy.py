import numpy as np
import bray
import torch

from .model import AtariModel
from .actor import AtariActor

bray.init(project="./atari-pengyao", trial="deploy")

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=({"image": torch.rand(1, 42, 42, 4, dtype=torch.float32)},),
    use_onnx="infer",
    local_mode=True,
)

remote_actor = bray.RemoteActor(port=8000, num_workers=6)

remote_actor.serve(
    Actor=AtariActor,
    remote_model=remote_model,
)

bray.run_until_asked_to_stop()
