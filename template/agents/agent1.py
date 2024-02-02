import numpy as np
import bray
import base64


def gae(trajectory: list, bootstrap_value=0.0):
    """Generalized Advantage Estimation"""
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


class Agent1(bray.Agent):
    def __init__(self, name, config: dict, state: bray.State):
        self.name = name

    async def on_tick(self, state: bray.State):
        obs = base64.b64decode(state.input["obs"])
        obs = np.frombuffer(
            obs,
            dtype=np.float32,
        ).reshape(42, 42, 4).transpose(2, 0, 1)
        obs = {"image": obs}
        reward = state.input["reward"]
        value, logit, action = await bray.forward("model1", obs)
        state.transition = {
            "state": obs,
            "action": action,
            "value": value,
            "logit": logit,
            "reward": np.array(np.sign(reward)),
            "raw_reward": reward,
        }
        state.output["action"] = action.tolist()

    async def on_episode(self, episode: list[bray.State], done: bool):
        trajectory = [s.transition for s in episode]
        last_transition = trajectory.pop()
        bootstrap_value = last_transition["value"]
        gae(trajectory, bootstrap_value)
        bray.push("buffer1", *trajectory)