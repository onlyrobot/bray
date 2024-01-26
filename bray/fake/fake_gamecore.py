import requests
import uuid
from threading import Thread
import time

config = {
    "fake_gamecore_step_start_data": b"fake_gamecore_step_start_data",
    "fake_gamecore_step_tick_data": b"fake_gamecore_step_tick_data",
    "fake_gamecore_step_stop_data": b"fake_gamecore_step_stop_data",
}
actor_url = "http://localhost:8000/step"


def actor_step(sess: requests.Session, session, step_kind, data):
    res = sess.post(
        actor_url,
        headers={
            "session": session,
            "step_kind": step_kind,
        },
        data=data,
    )
    if res.status_code != 200:
        raise Exception(res.text)
    return res


def rollout(session, config):
    sess = requests.Session()
    res = actor_step(
        sess,
        session,
        "start",
        config["fake_gamecore_step_start_data"],
    )
    print(res)
    for _ in range(1000000):
        import time

        time.sleep(0.5)
        res = actor_step(
            sess,
            session,
            "tick",
            config["fake_gamecore_step_tick_data"],
        )
        print(res)
    res = actor_step(
        sess,
        session,
        "stop",
        config["fake_gamecore_step_stop_data"],
    )
    print(res)


def endless_rollout(session, config):
    while True:
        try:
            rollout(session, config)
        except Exception as e:
            print(e)
            time.sleep(5)
        session = "game_" + str(uuid.uuid4())


for i in range(2):
    session = "game_" + str(uuid.uuid4())
    Thread(target=endless_rollout, args=(session, config)).start()
