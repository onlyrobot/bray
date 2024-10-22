from typing import Union
import asyncio

from bray.master.master import (
    get, merge, Worker, add_histogram,
)
from bray.master.master import set as bray_set
from bray.model.model import RemoteModel

import ray
import numpy as np


@ray.remote(num_cpus=0, name="model_pool", get_if_exists=True)
class ModelPool:
    def __init__(self, remote_model=None, pool=None, metric="win"):
        RemoteModel.max_cached_remote_model = 100000000
        self.name = ray.get_runtime_context().get_actor_name()
        self.remote_model, self.pool = remote_model, pool
        self.metric, self.all_names = metric, []
        self.pool_names = set([m.name for m in self.pool or []])
        self.explore_rate, self.pool_update_rate = 0.3, 0.2
        self.metrics, self.last_metrics = get("ModelPoolMetrics", {}), {}
        self.is_initialized = False
        if self.remote_model: asyncio.create_task(self.update())

    async def sample(self) -> RemoteModel:
        while self.remote_model and not self.is_initialized: 
            await asyncio.sleep(0.001)
        if self.pool: model = np.random.choice(self.pool)
        else:
            model = RemoteModel(await self.sample_name())
            model.current_name = model.name
        merge(f"{self.name}/{model.current_name}", 1, 
            desc={"time_window_cnt": "sample per minute"})
        return model

    async def sample_name(self) -> str:
        """从RemoteModel的检查点中采样得到一个模型，返回它的step"""
        if not self.metrics or np.random.random() < self.explore_rate:
            return np.random.choice(self.all_names)
        return np.random.choice(self.names, p=self.weights)

    async def update_pool(self, remote_model):
        model = RemoteModel(await self.sample_name())
        weights = await model.model.get_weights.remote(model.name)
        remote_model.current_name = model.name
        await remote_model.publish_weights(weights)

    def is_valid_name(self, name) -> bool:
        if name in self.all_names: return True
        if not name.startswith(self.remote_model.name): return False
        return name not in self.pool_names

    async def update(self, time_window=60):
        model = self.remote_model.model
        ckpt_steps = await model.get_ckpt_steps.remote()
        self.all_names = [f"{n}/clone-step-{s}" 
            for n, steps in ckpt_steps.items() for s in steps 
            if s and self.is_valid_name(n)]
        # self.all_names = list(set(self.all_names))
        metric_names = [
            f"{self.metric}/{name}" for name in self.all_names]
        master = Worker().master
        metrics = await master.batch_query.remote(metric_names)
        if self.last_metrics: self.metrics.clear()
        for n, m in zip(self.all_names, metrics):
            if not m.cnt: continue
            avg = m.sum / m.cnt
            last_avg = self.last_metrics.get(n, avg)
            avg = last_avg * 7 / 8 + avg / 8
            self.last_metrics[n] = self.metrics[n] = avg
        bray_set("ModelPoolMetrics", self.metrics)
        self.names = list(self.metrics.keys())
        weights = np.array(list(self.metrics.values()))
        weights -= self.metrics.get(self.remote_model.name, 0.5)
        weights = np.exp(100 * (1 - np.abs(weights)))
        self.weights = weights / np.sum(weights)
        for m in (self.pool or []): 
            if (self.is_initialized and 
            np.random.random() > self.pool_update_rate): continue
            await self.update_pool(m)
        self.is_initialized = True
        await asyncio.sleep(time_window)
        asyncio.create_task(self.update(time_window))


if __name__ == "__main__":
    from bray.model.test import AtariModel, forward_args
    import bray

    ray.init(namespace="model_pool", address="local")

    model = AtariModel()
    remote_model = RemoteModel(
        "model", model=model, forward_args=forward_args,
        checkpoint_interval=1,
    )
    weights = bray.get_torch_model_weights(model)

    for _ in range(10): 
        ray.get(remote_model.publish_weights(weights))

    model_pool = ModelPool.remote(remote_model, metric="win")
    print("Sample before:")
    print([ray.get(model_pool.sample.remote()).name for _ in range(10)])

    for i in range(10):
        merge(f"win/{remote_model.name}/clone-step-{i}", i / 10)
    ray.get(Worker().flush_and_reset_merge_interval())
    ray.get(model_pool.update.remote(time_window=1))
    print("Sample after:")
    print([ray.get(model_pool.sample.remote()).name for _ in range(10)])

    pool = [remote_model.clone(step=0, name=f"clone_{i}") for i in range(3)]
    model_pool = ModelPool.options(name="model_pool2").remote(
        remote_model, pool, metric="win")
    for _ in range(30):
        ray.get(model_pool.update.remote(time_window=1))
    print("Sample pool:")
    print([ray.get(model_pool.sample.remote()).current_name 
        for _ in range(10)])

    @ray.remote
    def update_weights():
        while True:
            m = ray.get(model_pool.sample.remote())
            m = RemoteModel(m.current_name)
            ray.get(m.publish_weights(weights))

    @ray.remote
    def update_metrics():
        while True:
            m = ray.get(model_pool.sample.remote())
            merge(f"win/{m.current_name}", np.random.random())

    update_weights.remote()
    update_metrics.remote()

    for _ in range(100000000):
        ray.get(model_pool.sample.remote()).current_name
        # print(ray.get(model_pool.sample.remote()).current_name, end=" ")

    print("Check the result for test")
    