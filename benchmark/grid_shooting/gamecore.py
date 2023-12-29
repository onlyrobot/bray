import requests
import uuid
import json
from threading import Thread
from .grid_shooting import GridShooting
import time

actor_url = "http://localhost:8000/step"


def actor_step(sess: requests.Session, game_id, step_kind, data):
    res = sess.post(
        actor_url,
        headers={
            "game_id": game_id,
            "step_kind": step_kind,
        },
        data=data,
    )
    if res.status_code != 200:
        raise Exception(res.text)
    return res.content


def rollout(game_id: str):
    sess = requests.Session()
    env = GridShooting()
    game_start_res = actor_step(sess, game_id, "start", b"")
    print(game_start_res)
    done, rewards = False, [0.0, 0.0]
    env.reset()
    while not done:
        state_0, extra_info_0 = env.get_state(0, None)
        state_1, extra_info_1 = env.get_state(1, None)
        data = {
            "obs": [state_0, state_1],
            "extra_info": [extra_info_0, extra_info_1],
            "rewards": rewards,
        }
        res = actor_step(sess, game_id, "tick", json.dumps(data))
        cmd = json.loads(res)["action"]
        rewards, done, _ = env.step(cmd[0], cmd[1])
    # final reward
    data = {"rewards": rewards}
    game_end_res = actor_step(sess, game_id, "stop", json.dumps(data))
    print(game_end_res)


def endless_rollout(game_id: str):
    while True:
        # rollout(game_id)
        try:
            rollout(game_id)
        except Exception as e:
            print(e)
        time.sleep(5)
        game_id = "game_" + str(uuid.uuid4())


# parallel rollout (may be thread or process)
for i in range(10):
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(game_id,)).start()
