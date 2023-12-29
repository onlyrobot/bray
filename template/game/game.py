import requests
import gym
import uuid
import json
from threading import Thread
from .atari_wrappers import wrap_deepmind
import time
import base64
import bray

actor_url = "http://localhost:8000/step"
# gym_id = "BreakoutNoFrameskip-v4"
gym_id = "BeamRiderNoFrameskip-v4"


def make_env(gym_id: str):
    env = gym.make(gym_id)
    env = wrap_deepmind(env)
    # print(env.action_space.n)
    # print(env.observation_space.shape)
    return env


def rollout(env, game_id: str):
    client = bray.Client("localhost", 8000)
    client.start()
    print("Game Start")
    done, reward = False, 0.0
    obs = env.reset()
    while not done:
        obs = base64.b64encode(obs.tobytes()).decode('utf-8')
        data = {"obs": obs, "reward": reward}
        res = client._tick(json.dumps(data))
        cmd = json.loads(res)
        cmd = int(cmd["action"])
        obs, reward, done, _ = env.step(cmd)
    # final reward
    obs = base64.b64encode(obs.tobytes()).decode('utf-8')
    data = {"obs": obs, "reward": reward}
    res = client._tick(json.dumps(data))
    client.stop()
    print("Game Stop")


def endless_rollout(gym_id: str, game_id: str):
    env = make_env(gym_id)
    while True:
        # rollout(gym_id, game_id)
        try:
            rollout(env, game_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
        time.sleep(5)
        game_id = "game_" + str(uuid.uuid4())


# parallel rollout (may be thread or process)
for i in range(4):
    game_id = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(gym_id, game_id)).start()
