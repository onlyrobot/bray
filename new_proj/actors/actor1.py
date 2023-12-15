import numpy as np
import bray
import json
from bray import NestedArray
import base64


def gae(trajectory, bootstrap_value=0.0):
    trajectory.append({"advantage": 0.0, "value": bootstrap_value})
    for i in reversed(range(len(trajectory) - 1)):
        t, next_t = trajectory[i], trajectory[i + 1]
        target_value = t["reward"] + 0.99 * next_t["value"]
        # 0.99: discount factor of the MDP
        delta = target_value - t["value"]
        # 0.95: discount factor of the gae
        advantage = delta + 0.99 * 0.95 * next_t["advantage"]
        t["advantage"] = np.array(advantage, dtype=np.float32)
    trajectory.pop(-1)  # drop the fake


class Actor(bray.Actor):
    def __init__(self):
        self.remote_model = bray.RemoteModel("model1")
        self.remote_buffer = bray.RemoteBuffer("buffer1")
        self.reward_metric = bray.Metric(
            name="reward",
            up_bound=1000,
            low_bound=0,
            max_samples=1000,
            print_report=True,
        )
        self.value_metric = bray.Metric("value")
        self.logit_metric = bray.Metric("logit")

    async def start(self, game_id, data: bytes) -> bytes:
        self.game_id = game_id
        self.trajectory = []
        self.episode_reward = 0.0
        bray.logger.info(f"Actor.start: {game_id}")
        return b"Game started."

    async def tick(self, data: bytes) -> bytes:
        data = json.loads(data)
        obs = base64.b64decode(data["obs"])
        obs = np.frombuffer(obs, dtype=np.float32).reshape(42, 42, 4)
        reward = data["reward"]
        self.reward_metric.merge(reward)
        self.episode_reward += reward
        state = {"image": obs}
        value, logit, action = await self.remote_model.forward(state)
        self.logit_metric.merge(logit)
        self.value_metric.merge(value)
        self._append_to_trajectory(state, action, reward, value, logit)
        return json.dumps({"action": action.tolist()}).encode()

    async def end(self, data: bytes) -> bytes:
        data = json.loads(data)
        reward = data["reward"]
        self.episode_reward += reward
        bray.merge("episode_reward", self.episode_reward)
        self._append_to_trajectory(None, None, reward, None, None, end=True)
        bray.logger.info(f"Actor.end: {self.game_id}")
        return b"Game ended."

    def _append_to_trajectory(
        self,
        state: NestedArray,
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
        if end or len(self.trajectory) >= 128:
            gae(
                self.trajectory,
                bootstrap_value=0.0 if end else value,
            )
            self.remote_buffer.push(*self.trajectory)
            self.trajectory.clear()
        if end:
            return
        transition = {
            "state": state,
            "action": action,
            "reward": np.array(0.0, dtype=np.float32),
            "value": value,
            "logit": logit,
        }
        self.trajectory.append(transition)
