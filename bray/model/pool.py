import asyncio

from bray.master.master import set, get, merge, Worker

import ray
import numpy as np


@ray.remote(num_cpus=0, name="ModelPool", get_if_exists=True)
class ModelPool:
    def __init__(self, remote_model = None, pool = None, metric="win"):
        self.remote_model, self.pool = remote_model, pool
        self.ckpt_steps = ray.get(
            remote_model.model.get_ckpt_steps.remote(remote_model.name))
        self.metrics = get("ModelPoolMetrics", {})
        self.explore_rate = 0.3
        asyncio.create_task(self.update())

    async def sample(self) -> int:
        """从RemoteModel的检查点中采样得到一个模型，返回它的step"""
        if not self.metrics or np.random.random() < self.explore_rate:
            if not self.ckpt_steps:
                return 0
            return np.random.choice(self.ckpt_steps)
        return np.random.choice(self.steps, p=self.weights)

    async def update(self, time_window=60):
        model = self.remote_model.model
        self.ckpt_steps = await model.get_ckpt_steps.remote(
            self.remote_model.name)
        names = [
            f"win/{self.remote_model.name}/clone-step-{n}" 
            for n in self.ckpt_steps
        ]
        metrics = await Worker().master.batch_query.remote(names)
        self.metrics.clear()
        for n, m in zip(self.ckpt_steps, metrics):
            if not m.cnt:
                continue
            self.metrics[n] = m.sum / m.cnt - 0.5
        set("ModelPoolMetrics", self.metrics)
        self.steps = list(self.metrics.keys())
        weights = list(self.metrics.values())
        weights = np.exp(100 * np.array(weights))
        self.weights = weights / np.sum(weights)
        await asyncio.sleep(time_window)
        asyncio.create_task(self.update(time_window))


if __name__ == "__main__":
    ray.init(namespace="model_pool", address="local")

    model_pool = ModelPool.remote(None, metric="win")
    print("Sample before:")
    print([ray.get(model_pool.sample.remote()) for _ in range(10)])

    for i in range(10):
        merge(f"win/fake_model-{i}", i)
    ray.get(Worker().flush_and_reset_merge_interval())
    ray.get(model_pool.update.remote())
    print("Sample after:")
    print([ray.get(model_pool.sample.remote()) for _ in range(10)])
    print("Check the result for test")
    