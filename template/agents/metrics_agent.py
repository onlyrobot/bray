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
    def __init__(self, name, config: dict, state: bray.State):
        self.name = name
        self.need_metrics = random.random() < 0.01
        self.episode_reward = 0.0
        self.reward_metric = bray.Metric("reward")
        self.value_metric = bray.Metric("value")
        self.logit_metric = bray.Metric("logit")

    async def on_tick(self, state: bray.State):
        if not self.need_metrics:
            return
        transition = await state.wait("transition")
        reward = transition["reward"]
        value = transition["value"]
        logit = transition["logit"]
        self.episode_reward += transition["raw_reward"]
        if state.actor != 0:
            return
        self.reward_metric.merge(reward)
        self.value_metric.merge(value)
        self.logit_metric.merge(logit)

    async def on_episode(self, episode: list[State], done: bool):
        if not done or not episode:
            return
        bray.merge(f"episode_reward/{self.name}", self.episode_reward)
