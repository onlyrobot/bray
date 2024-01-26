import requests
import gym
import uuid
import json
from threading import Thread
from .atari_wrappers import wrap_deepmind
import time

import socket
import struct

gym_id = "BeamRiderNoFrameskip-v4"


def actor_step(sock: socket.socket, session, step_kind, data: bytes):
    session, step_kind = session.encode(), step_kind.encode()
    header = struct.pack(
        "!6q",
        len(session),
        len(step_kind),
        0,  # len(key)
        0,  # len(token)
        len(data),  # len(data)
        0,  # time
    )
    sock.sendall(
        header + session + step_kind + b"" + b"" + data,  # key and token is empty
    )
    data = sock.recv(8 * 3, socket.MSG_WAITALL)
    session_size, body_size, time = struct.unpack("!3q", data)
    data = sock.recv(session_size + body_size, socket.MSG_WAITALL)
    assert data[:session_size] == session and time == 0
    return data[session_size:]


def make_env(gym_id: str):
    env = gym.make(gym_id)
    env = wrap_deepmind(env)
    return env


def rollout(gym_id: str, session: str):
    # 创建一个socket对象
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到目标主机
    host, port = "127.0.0.1", 8000
    sock.connect((host, port))
    env = make_env(gym_id)
    game_start_res = actor_step(sock, session, "start", b"")
    print(game_start_res)
    done, reward = False, 0.0
    obs = env.reset()
    while not done:
        data = {"obs": obs.tolist(), "reward": reward}
        res = actor_step(
            sock,
            session,
            "tick",
            json.dumps(data).encode(),
        )
        cmd = json.loads(res)
        cmd = int(cmd["action"])
        obs, reward, done, _ = env.step(cmd)
    # final reward
    data = {"reward": reward}
    game_end_res = actor_step(
        sock,
        session,
        "stop",
        json.dumps(data).encode(),
    )
    print(game_end_res)


def endless_rollout(gym_id: str, session: str):
    while True:
        # rollout(gym_id, session)
        try:
            rollout(gym_id, session)
        except Exception as e:
            print(e)
        time.sleep(5)
        session = "game_" + str(uuid.uuid4())


# parallel rollout (may be thread or process)
for i in range(4):
    session = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(gym_id, session)).start()
