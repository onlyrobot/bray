import requests
import gym
import uuid
import json
from threading import Thread
from .atari_wrappers import wrap_deepmind
import time

actor_url = "http://localhost:8000/step"
gym_id = "BeamRiderNoFrameskip-v4"


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
    return res.content


def make_env(gym_id: str):
    env = gym.make(gym_id)
    env = wrap_deepmind(env)
    return env


def rollout(gym_id: str, game_id: str):
    env = make_env(gym_id)
    game_start_res = actor_step(game_id, "start", b"")
    print(game_start_res)
    done, reward = False, 0.0
    obs = env.reset()
    while not done:
        data = {"obs": obs.tolist(), "reward": reward}
        res = actor_step(game_id, "tick", json.dumps(data))
        cmd = json.loads(res)
        cmd = int(cmd["action"])
        obs, reward, done, _ = env.step(cmd)
    # final reward
    data = {"reward": reward}
    game_end_res = actor_step(game_id, "end", json.dumps(data))
    print(game_end_res)


def endless_rollout(gym_id: str, game_id: str):
    while True:
        # rollout(gym_id, game_id)
        try:
            rollout(gym_id, game_id)
        except Exception as e:
            print(e)
        time.sleep(5)
        game_id = "game_" + str(uuid.uuid4())


# parallel rollout (may be thread or process)
for i in range(8):
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(gym_id, game_id)).start()