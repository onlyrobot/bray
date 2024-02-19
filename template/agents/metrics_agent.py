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
        self.need_metric = random.random() < 1 / 2 ** state.actor
        self.episode_reward = 0.0
        self.reward_metric = bray.Metric("reward")
        self.value_metric = bray.Metric("value")
        self.logit_metric = bray.Metric("logit")

    async def on_tick(self, state: bray.State):
        transition = await state.wait("transition")
        self.episode_reward += transition["raw_reward"]
        if not self.need_metric:
            return
        self.reward_metric.merge(transition["reward"])
        self.value_metric.merge(transition["value"])
        self.logit_metric.merge(transition["logit"])

    async def on_episode(self, episode: list[State], done: bool):
        if not done or not episode:
            return
        bray.merge(f"episode_reward/{self.name}", self.episode_reward)
