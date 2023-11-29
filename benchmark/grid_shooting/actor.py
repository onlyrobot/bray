import numpy as np
import bray
import json
from bray import NestedArray
import random


def gae(trajectory: list[NestedArray], bootstrap_value):
    trajectory.append({"advantage": 0.0, "value": bootstrap_value})
    for i in reversed(range(len(trajectory) - 1)):
        t, next_t = trajectory[i], trajectory[i + 1]
        # 0.99: discount factor of the MDP
        delta = t["reward"] + 0.99 * next_t["value"] - t["value"]
        # 0.95: discount factor of the gae
        advantage = delta + 0.99 * 0.95 * next_t["advantage"]
        t["advantage"] = np.array(advantage, dtype=np.float32)
    trajectory.pop(-1)  # drop the fake


class GridShootingActor(bray.Actor):
    def __init__(
        self,
        remote_model: bray.RemoteModel,
        remote_buffer: bray.RemoteBuffer = None,
        target_model_max_reuse=100,
    ):
        self.remote_model = remote_model
        self.remote_buffer = remote_buffer
        self.target_model = None
        self.target_model_reuse = 0
        self.target_model_max_reuse = target_model_max_reuse

    def start(self, game_id, data: bytes) -> bytes:
        self.game_id = game_id
        self.trajectory = []
        self.episode_reward = 0.0
        if (
            not self.target_model
            or self.target_model_reuse > self.target_model_max_reuse
        ):
            self.target_model = self.remote_model.clone(
                step=random.randint(0, self.remote_model.step)
            )
            self.target_model_reuse = 0
        else:
            self.target_model_reuse += 1
        bray.logger.info(f"Actor.start: {game_id}")
        return b"Game started."

    async def tick(self, data: bytes) -> bytes:
        data = json.loads(data)
        state_0, state_1 = np.array(data["obs"], dtype=np.float32)
        reward_0, _ = data["rewards"]
        self.episode_reward += reward_0
        extra_info_0, extra_info_1 = data["extra_info"]

        def build_action_mask(legal_actions):
            if not legal_actions:
                return np.ones(9, dtype=np.float32)
            action_mask = np.zeros(9, dtype=np.float32)
            for action in legal_actions:
                action_mask[int(action)] = 1.0
            return action_mask

        action_mask_0 = build_action_mask(extra_info_0["legal_actions"])
        obs = {"obs": state_0, "action_mask": action_mask_0}
        value, logit, action = await self.remote_model.forward(obs)
        self._append_to_trajectory(obs, action, reward_0, value, logit)
        action_mask_1 = build_action_mask(extra_info_1["legal_actions"])
        _, _, action_1 = await self.target_model.forward(
            {"obs": state_1, "action_mask": action_mask_1}
        )
        return json.dumps({"action": [int(action), int(action_1)]}).encode()

    def end(self, data: bytes) -> bytes:
        data = json.loads(data)
        reward_0, _ = data["rewards"]
        self.episode_reward += reward_0
        bray.merge("episode_reward", self.episode_reward)
        bray.merge("episode_reward", self.episode_reward, target=self.target_model.name)
        self._append_to_trajectory(None, None, reward_0, None, None, end=True)
        bray.logger.info(f"Actor.end: {self.game_id}")
        return b"Game ended."

    def _append_to_trajectory(
        self,
        obs: NestedArray,
        action: NestedArray,
        reward: float,
        value: NestedArray,
        logit: NestedArray,
        end=False,
    ):
        if not self.remote_buffer:
            return
        # clip reward
        reward = np.array(reward, dtype=np.float32).clip(-100, 100)
        if len(self.trajectory) > 0:
            self.trajectory[-1]["reward"] = reward
        if end or len(self.trajectory) > 128:
            gae(
                self.trajectory,
                bootstrap_value=0.0 if end else self.trajectory[-1]["value"],
            )
            self.remote_buffer.push(*self.trajectory)
            self.trajectory = []
        transition = {
            "obs": obs,
            "action": action,
            "reward": np.array(0.0, dtype=np.float32),
            "value": value,
            "logit": logit,
        }
        self.trajectory.append(transition)
