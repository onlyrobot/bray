import bray

class FakeActor(bray.Actor):
    def __init__(self, agents, config, game_id, data):
        self.agents = agents
        print("FakeActor.__init__: ", agents, config, game_id, data)

    def tick(self, data):
        print("FakeActor.tick: ", data)
        return self.config["fake_actor_tick_return"]

    def end(self, data):
        print("FakeActor.end: ", data)
        return self.config["fake_actor_end_return"]
    
config = {
    "fake_actor_tick_return": b"fake_actor_tick_return",
    "fake_actor_end_return": b"fake_actor_end_return"
}
actor_port = 8000

remote_actor = bray.RemoteActor(FakeActor, None, config)
remote_actor.serve_background(actor_port)

# wait for SIGTERM or SIGINT (i.e. Ctrl+C) to stop the actor
import signal
signal.sigwait([signal.SIGTERM, signal.SIGINT])