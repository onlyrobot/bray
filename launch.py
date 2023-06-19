import bray

class MyModel:
    def __init__(self):
        print("Model.__init__")

    def __call__(self, state):
        print("Model forward")
        return "action", "value"
    
    def get_weights(self):
        return "weights"
    
    def set_weights(self, weights, version):
        print(f"Model.set_weights: {weights}, {version}")




class MyTrainer:
    def __init__(self):
        pass

    def train(self, model, replays):
        version = 0
        for replay in replays:
            print("Trainer.train")
            print("replay", replay)
            import time
            time.sleep(1)
            version += 1
            remote_model.publish_weights(1, version)

class MyActor:
    def __init__(self, agents, config, game_id, data):
        self.agents = agents
        print("Actor.__init__: ", agents, config, game_id, data)

    def tick(self, round_id, data):
        state = data
        agents = self.agents["default"]
        action, value = agent.remote_model.forward(state)
        agent.remote_buffer.push((state, action, value))
        print("Actor.step: ", round_id, data)
        return data

    def end(self, round_id, data):
        print("Actor.end: ", round_id, data)
        return data

    
remote_model = bray.RemoteModel("model1", MyModel())

remote_trainer = bray.RemoteTrainer(MyTrainer())

remote_buffer = remote_trainer.new_buffer("buffer1")

agent = bray.Agent(remote_model, remote_buffer)

remote_actor = bray.RemoteActor(MyActor, {"default": agent}, None)
remote_actor.serve_background()

remote_trainer.train(remote_model, remote_buffer)