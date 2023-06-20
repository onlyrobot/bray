import requests
import uuid
from threading import Thread

config = {
    "fake_gamecore_step_start_data": b"fake_gamecore_step_start_data",
    "fake_gamecore_step_tick_data": b"fake_gamecore_step_tick_data",
    "fake_gamecore_step_end_data": b"fake_gamecore_step_end_data",
}
actor_url = "http://localhost:8000/step"


def actor_step(game_id, round_id, step_kind, data):
    res = requests.post(
        actor_url,
        headers={
            "game_id": game_id,
            "round_id": str(round_id),
            "step_kind": step_kind,
        },
        data=data,
    )
    if res.status_code != 200:
        raise Exception(res.text)
    return res.json()


def rollout(game_id, config):
    res = actor_step(
        game_id,
        0,
        "start",
        config["fake_gamecore_step_start_data"],
    )
    print(res)
    for i in range(1000000):
        import time

        time.sleep(0.5)
        res = actor_step(
            game_id,
            i,
            "tick",
            config["fake_gamecore_step_tick_data"],
        )
        print(res)
    res = actor_step(
        game_id,
        0,
        "end",
        config["fake_gamecore_step_end_data"],
    )
    print(res)


for i in range(2):
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=rollout, args=(game_id, config)).start()
