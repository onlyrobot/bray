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
    num_workers=2,
)
eval_buffer = bray.RemoteBuffer(
    "eval_buffer",
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
    train_buffer=train_buffer,
    eval_buffer=eval_buffer,
    batch_size=8,
    weights_publish_interval=1,
    num_steps=1000000,
)

cpu_num = int(sum([node["Resources"].get("CPU", 0) for node in ray.nodes()]))

fake_data = {
    "image": np.random.randn(42, 42, 4).astype(np.float32),
    "label": np.array(0.0, dtype=np.float32),
}
eval_source = [AtariDataset(fake_data) for _ in range(cpu_num * 2)]
train_source = [AtariDataset(fake_data) for _ in range(cpu_num * 2)]

EPOCH = 8
eval_buffer.add_source(*eval_source)

for i in range(EPOCH):
    print(f"Epoch {i} start")
    ret = train_buffer.add_source(*train_source)
    ray.get(ret)
    print(f"Epoch {i} done")

time.sleep(60)  # wait for all data to be consumed

print(f"Train all {EPOCH} epoch done!")
