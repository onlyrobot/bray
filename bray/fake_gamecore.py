import requests
import uuid
from threading import Thread
import time

config = {
    "fake_gamecore_step_start_data": b"fake_gamecore_step_start_data",
    "fake_gamecore_step_tick_data": b"fake_gamecore_step_tick_data",
    "fake_gamecore_step_end_data": b"fake_gamecore_step_end_data",
}
actor_url = "http://localhost:8000/step"


def actor_step(game_id, step_kind, data):
    res = requests.post(
        actor_url,
        headers={
            "game_id": game_id,
            "step_kind": step_kind,
        },
        data=data,
    )
    if res.status_code != 200:
        raise Exception(res.text)
    return res


def rollout(game_id, config):
    res = actor_step(
        game_id,
        "start",
        config["fake_gamecore_step_start_data"],
    )
    print(res)
    for _ in range(1000000):
        import time

        time.sleep(0.5)
        res = actor_step(
            game_id,
            "tick",
            config["fake_gamecore_step_tick_data"],
        )
        print(res)
    res = actor_step(
        game_id,
        "end",
        config["fake_gamecore_step_end_data"],
    )
    print(res)


def endless_rollout(game_id, config):
    while True:
        try:
            rollout(game_id, config)
        except Exception as e:
            print(e)
            time.sleep(5)
        game_id = "game_" + str(uuid.uuid4())


for i in range(2):
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(game_id, config)).start()
