import requests
from threading import Thread


actor_url = "http://localhost:8000/step"


def step(game_id, round_id, kind, data):
    return requests.post(
        actor_url,
        json={
            "game_id": game_id,
            "round_id": round_id,
            "kind": kind,
            "data": data,
        },
    ).json()


def rollout(game_id, data):
    print(
        step(
            game_id,
            0,
            "start",
            "",
        )
    )
    for i in range(1000000):
        import time

        time.sleep(0.5)
        print(
            step(
                game_id,
                i,
                "tick",
                data,
            )
        )
    print(
        step(
            game_id,
            0,
            "end",
            "",
        )
    )


for i in range(2):
    Thread(target=rollout, args=("game: " + str(i), "hello")).start()
