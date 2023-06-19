import requests
import gym
from threading import Thread
# from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from .atari_wrappers import wrap_deepmind

actor_url = "http://localhost:8000/step"
gym_id = "BeamRiderNoFrameskip-v4"


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


def make_env(gym_id: str):
    env = gym.make(gym_id)
    env = wrap_deepmind(env)
    return env


def rollout(gym_id: str, game_id: str):
    env = make_env(gym_id)
    game_start_res = step(game_id, 0, "start", "")
    print(game_start_res)
    done, round_id, reward = False, 0, 0.0
    obs = env.reset()
    while not done:
        data = {"obs": obs, "reward": reward}
        cmd = step(game_id, round_id, "tick", data)
        round_id += 1
        obs, reward, done, _ = env.step(cmd)
    data = {"reward": reward}
    game_end_res = step(game_id, 0, "end", data)
    print(game_end_res)


# parallel rollout (may be thread or process)
for i in range(1):
    import uuid
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=rollout, args=(gym_id, str(uuid.uuid4()))).start()
