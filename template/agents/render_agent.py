import numpy as np
import bray
from bray.actor.agent import State
import random


class RenderAgent(bray.Agent):
    """
    定时渲染游戏画面，并输出到Tensorboard页面，为了避免过于频繁，
    随机选择部分episode进行渲染
    """

    def __init__(self, name, config: dict):
        self.name = name
        self.need_render = random.random() < 0.01
        # 额外保存一下episode的obs，用于生成完整游戏视频
        self.episode_obs = []

    async def on_tick(self, state: bray.State):
        if not self.need_render:
            return
        input = await state.input
        self.episode_obs.append(input["obs"])

    async def on_episode(self, episode: list[State], done: bool):
        if not done or not self.episode_obs:
            return
        obs = np.array([self.episode_obs])
        bray.add_video(f"render/{self.name}", obs)