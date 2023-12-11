import numpy as np
import time
import ray
import bray

from ..model import AtariModel
from .dataset import AtariDataset
from .trainer import train_atari

bray.init(project="./atari-pengyao", trial="sl-v0")

remote_model = bray.RemoteModel(
    name="atari_model",
    model=AtariModel(),
    forward_args=({"image": np.random.randn(42, 42, 4).astype(np.float32)},),
    num_workers=0,
    local_mode=True,
    use_onnx="infer",
)
bray.set_tensorboard_step(remote_model.name)

train_buffer = bray.RemoteBuffer(
    "train_buffer",
    size=128,
    batch_size=8,
    num_workers=1,
)
eval_buffer = bray.RemoteBuffer(
    "eval_buffer",
    size=128,
    batch_size=8,
    num_workers=1,
)

remote_trainer = bray.RemoteTrainer(
    use_gpu=None,
    num_workers=None,
)

remote_trainer.train(
    train=train_atari,
    remote_model=remote_model,
    train_buffer=train_buffer,
    eval_buffer=eval_buffer,
    batch_size=8,
    weights_publish_interval=1,
    num_steps=1000000,
)

fake_data = {
    "image": np.random.randn(42, 42, 4).astype(np.float32),
    "label": np.array(0.0, dtype=np.float32),
}
eval_source = [AtariDataset(fake_data) for _ in range(30)]
train_source = [AtariDataset(fake_data) for _ in range(30)]

EPOCH = 8
ret = train_buffer.add_source(*train_source, epoch=8)
eval_buffer.add_source(*eval_source, epoch=100000, num_workers=4)

print(ray.get(ret))

time.sleep(10)  # wait for all data to be consumed

print(f"Train all {EPOCH} epoch done!")
