import numpy as np
import bray
import json
from bray import NestedArray


def gae(trajectory: list[NestedArray]) -> None:
    if len(trajectory) == 0:
        return []
    trajectory[-1]["advantage"] = np.array(0.0, dtype=np.float32)
    for i in reversed(range(len(trajectory) - 1)):
        t, next_t = trajectory[i], trajectory[i + 1]
        # 0.99: discount factor of the MDP
        delta = t["reward"] + 0.99 * next_t["value"] - t["value"]
        # 0.95: discount factor of the gae
        advantage = delta + 0.99 * 0.95 * next_t["advantage"]
        t["advantage"] = np.array(advantage, dtype=np.float32)


class AtariActor(bray.Actor):
    def __init__(self, model: str, buffer: str = None):
        self.remote_model = bray.RemoteModel(name=model)
        self.remote_buffer = None
        if buffer:
            self.remote_buffer = bray.RemoteBuffer(name=buffer)

    def start(self, game_id, data: bytes) -> bytes:
        self.game_id = game_id
        self.trajectory = []
        self.episode_reward = 0.0
        print("Actor.start: ", game_id)
        return b"Game started."

    async def tick(self, data: bytes) -> bytes:
        data = json.loads(data)
        obs = np.array(data["obs"], dtype=np.float32)
        reward = data["reward"]
        self.episode_reward += reward
        value, logit, action = await self.remote_model.forward(obs)
        self._append_to_trajectory(obs, action, reward, value, logit)
        return json.dumps({"action": action.tolist()})

    def end(self, data: bytes) -> bytes:
        data = json.loads(data)
        reward = data["reward"]
        self.episode_reward += reward
        bray.merge("episode_reward", self.episode_reward)
        self._append_to_trajectory(None, None, reward, None, None, end=True)
        print("Actor.end: ", self.game_id)
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
        reward = np.array(np.sign(reward), dtype=np.float32)
        if len(self.trajectory) > 0:
            self.trajectory[-1]["reward"] = reward
        if end or len(self.trajectory) > 128:
            gae(self.trajectory)
            self.remote_buffer.push(*self.trajectory)
            self.trajectory = []
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "value": value,
            "logit": logit,
        }
        self.trajectory.append(transition)
