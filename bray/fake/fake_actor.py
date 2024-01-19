import bray

bray.init(project="./fake-actor", trial="deploy")


class FakeActor(bray.Actor):
    def __init__(self, config):
        self.config = config
        print("FakeActor.__init__: ", config)

    async def start(self, game_id, data):
        print("FakeActor.start: ", game_id, data)
        return self.config["fake_actor_start_return"]

    async def tick(self, data):
        print("FakeActor.tick: ", data)
        return self.config["fake_actor_tick_return"]

    async def stop(self, data):
        print("FakeActor.stop: ", data)
        return self.config["fake_actor_stop_return"]


config = {
    "fake_actor_start_return": b"fake_actor_start_return",
    "fake_actor_tick_return": b"fake_actor_tick_return",
    "fake_actor_stop_return": b"fake_actor_stop_return",
}
actor_port = 8000

remote_actor = bray.RemoteActor(port=actor_port, use_tcp=True)
remote_actor.serve(Actor=FakeActor, config=config)

bray.run_until_asked_to_stop()
