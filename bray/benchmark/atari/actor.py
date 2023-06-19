import numpy as np
import bray

class AtariActor(bray.Actor):
    def __init__(self, agents, config, game_id, data):
        self.agents = agents
        print("Actor.__init__: ", agents, config, game_id, data)

    def _append_to_trajectory(self, trajectory, end=False):
        agent = self.agents["default"]
        # agent.remote_buffer.push(replay)
        pass

    def tick(self, round_id, data):
        print("Actor.tick: ", round_id, data)
        agent = self.agents["default"]
        obs, reward = data["obs"], data["reward"]
        action, value = agent.remote_model.forward(obs)
        self._append_to_trajectory((obs, action, reward, value))
        print("Actor.step: ", round_id, data)
        return {"action": action}

    def end(self, round_id, data):
        reward = data["reward"]
        self._append_to_trajectory((None, None, reward, None), end=True)
        print("Actor.end: ", round_id, data)
        return "Game ended."