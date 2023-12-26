import numpy as np
import bray
from bray.actor.agent import State
import random


class MetricsAgent(bray.Agent):
    """
    指标收集、处理、统计的Agent，一般将指标输出到Tensorboard页面，
    并且在检测到异常值时，输出到日志文件，为了避免过于频繁，
    随机选择部分episode进行渲染
    """

    reward_metric = bray.Metric("reward")
    value_metric = bray.Metric("value")
    logit_metric = bray.Metric("logit")

    def __init__(self, name: str):
        self.name = name
        self.need_metrics = random.random() < 0.01
        self.trajectory = []
        self.episode_reward = 0.0

    async def on_tick(self, state: bray.State):
        if not self.need_metrics:
            return
        transition = await state.transition
        self.trajectory.append(transition)
        reward = transition["reward"]
        value = transition["value"]
        logit = transition["logit"]
        self.episode_reward += transition["raw_reward"]
        MetricsAgent.reward_metric.merge(reward)
        MetricsAgent.value_metric.merge(value)
        MetricsAgent.logit_metric.merge(logit)
        bray.add_histogram(f"logit/{self.name}", logit)

    async def on_episode(self, episode: list[State], done: bool):
        if not done or not self.trajectory:
            return
        rewards = np.array([t["reward"] for t in self.trajectory])
        bray.add_histogram(f"reward/{self.name}", rewards)
        values = np.array([t["value"] for t in self.trajectory])
        bray.add_histogram(f"value/{self.name}", values)
        bray.merge(f"episode_reward/{self.name}", self.episode_reward)
