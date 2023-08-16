import bray

bray.init(project="./fake-actor", trial="deploy")


class FakeActor(bray.Actor):
    def __init__(self, config):
        self.config = config
        print("FakeActor.__init__: ", config)

    def start(self, game_id, data):
        print("FakeActor.start: ", game_id, data)
        return self.config["fake_actor_start_return"]

    async def tick(self, data):
        print("FakeActor.tick: ", data)
        return self.config["fake_actor_tick_return"]

    def end(self, data):
        print("FakeActor.end: ", data)
        return self.config["fake_actor_end_return"]


config = {
    "fake_actor_start_return": b"fake_actor_start_return",
    "fake_actor_tick_return": b"fake_actor_tick_return",
    "fake_actor_end_return": b"fake_actor_end_return",
}
actor_port = 8000

remote_actor = bray.RemoteActor(port=actor_port, use_tcp=False)
remote_actor.serve(Actor=FakeActor, config=config)

bray.run_until_asked_to_stop()
