import requests
import gym
from threading import Thread
from .atari_wrappers import wrap_deepmind

actor_url = "http://localhost:8000/step"
gym_id = "BeamRiderNoFrameskip-v4"


def step(game_id, round_id, kind, data):
    import pickle, base64
    data = base64.b64encode(pickle.dumps(data)).decode("utf-8")
    res = requests.post(
        actor_url,
        json={
            "game_id": game_id,
            "round_id": round_id,
            "kind": kind,
            "data": data,
        },
    )
    if res.status_code != 200:
        raise Exception(res.text)
    res = res.json()
    return pickle.loads(base64.b64decode(res["data"]))


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
    print(obs.shape)
    while not done:
        data = {"obs": obs, "reward": reward}
        cmd = step(game_id, round_id, "tick", data)
        round_id += 1
        cmd = int(cmd["action"])
        obs, reward, done, _ = env.step(cmd)
    data = {"reward": reward}
    game_end_res = step(game_id, 0, "end", data)
    print(game_end_res)


# parallel rollout (may be thread or process)
for i in range(10):
    import uuid
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=rollout, args=(gym_id, game_id)).start()
