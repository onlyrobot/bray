import requests
from threading import Thread


url = "http://localhost:8000/step"

def step(game_id, data):
    print(
        requests.post(
            url,
            json={
                "game_id": game_id,
                "round_id": 0,
                "kind": "start",
                "data": data,
            },
        ).json()
    )
    for i in range(100):
        print(
            requests.post(
                url,
                json={
                    "game_id": game_id,
                    "round_id": i,
                    "kind": "tick",
                    "data": data,
                },
            ).json()
        )
    print(
        requests.post(
            url,
            json={
                "game_id": game_id,
                "round_id": 0,
                "kind": "end",
                "data": data,
            },
        ).json()
    )


for i in range(10):
    Thread(target=step, args=("game: " + str(i), "hello")).start()
