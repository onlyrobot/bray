import numpy as np
import bray
import json
from bray import NestedArray


def gae(trajectory: list[NestedArray]) -> list[NestedArray]:
    if len(trajectory) == 0:
        return []
    trajectory[-1]["advantage"] = np.array(0.0, dtype=np.float32)
    for i in reversed(range(len(trajectory) - 1)):
        t, next_t = trajectory[i], trajectory[i + 1]
        # 0.99: discount factor of the MDP
        delta = t["reward"] + 0.99 * next_t["value"] - t["value"]
        # 0.95: discount factor of the gae
        advantage = delta + 0.99 * 0.95 * next_t["advantage"]
        t["advantage"] = advantage
    return trajectory


class AtariActor(bray.Actor):
    def __init__(self, agents, config, game_id, data: bytes):
        self.agents, self.config = agents, config
        self.game_id = game_id
        self.trajectory = []
        print("Actor.start: ", game_id)

    def _append_to_trajectory(
        self,
        obs: NestedArray,
        action: NestedArray,
        reward: float,
        value: NestedArray,
        logit: NestedArray,
        end=False,
    ):
        agent = self.agents["agent1"]
        if not agent.remote_buffer:
            return
        # clip reward
        reward = np.sign(np.array(reward, dtype=np.float32))
        if len(self.trajectory) > 0:
            self.trajectory[-1]["reward"] = reward
        if end or len(self.trajectory) > 10:
            self.trajectory = gae(self.trajectory)
            for transition in self.trajectory:
                agent.remote_buffer.push(transition)
            self.tranjectory = []
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "value": value,
            "logit": logit,
        }
        self.trajectory.append(transition)

    def tick(self, data: bytes) -> bytes:
        data = json.loads(data)
        agent = self.agents["agent1"]
        obs = np.array(data["obs"], dtype=np.float32)
        reward = data["reward"]
        value, logit, action = agent.remote_model.forward(obs)
        self._append_to_trajectory(obs, action, reward, value, logit)
        return json.dumps({"action": action.tolist()})

    def end(self, data: bytes) -> bytes:
        data = json.loads(data)
        reward = data["reward"]
        self._append_to_trajectory(None, None, reward, None, None, end=True)
        print("Actor.end: ", self.game_id)
        return b"Game ended."
