import numpy as np
import bray

class AtariActor(bray.Actor):
    def __init__(self, agents, config, game_id, data):
        self.agents = agents
        print("Actor.__init__: ", agents, config, game_id, data)

    def _append_to_trajectory(self, trajectory, end=False):
        agent = self.agents["agent1"]
        # agent.remote_buffer.push(replay)
        pass

    def tick(self, round_id, data):
        print("Actor.tick: ", round_id)
        agent = self.agents["agent1"]
        obs, reward = data["obs"], data["reward"]
        value, logit, action = agent.remote_model.forward(obs)
        self._append_to_trajectory((obs, action, reward, value, logit))
        print("Actor.step: ", round_id)
        return {"action": action}

    def end(self, round_id, data):
        reward = data["reward"]
        self._append_to_trajectory((None, None, reward, None, None), end=True)
        print("Actor.end: ", round_id)
        return "Game ended."