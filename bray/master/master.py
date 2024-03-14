import time
import asyncio
import ray


class Master:
    def __init__(self, time_window):
        self.registery, self.data = {}, {}
        self.msgs = []
        trial_path = ray.get_runtime_context().namespace
        self.log_path = f"{trial_path}/bray.log"
        self.time_window = time_window
        self.flush_cond = asyncio.Condition()
        asyncio.create_task(self.flush_log())

    async def push(self, key: str, value: object):
        self.data[key] = value

    def get(self, key: str) -> object:
        return self.data[key]

    def register(self, key: str) -> int:
        id = self.registery.get(key, 0)
        self.registery[key] = id + 1
        return id
    
    async def log(self, msg: str, flush: bool=False):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.msgs.append(f"{timestamp} {msg}\n")
        if not flush:
            return
        async with self.flush_cond:
            self.flush_cond.notify()

    def _flush_log(self):
        self.msgs, msgs = [], self.msgs
        with open(self.log_path, "+a") as f:
            f.writelines(msgs)
    
    async def flush_log(self):
        async with self.flush_cond:
            try:
                await asyncio.wait_for(
                    self.flush_cond.wait(), self.time_window,
                )
            except asyncio.TimeoutError:
                pass
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._flush_log,
            )
        except Exception as e:
            print(f"Log to {self.log_path} error: {e}")
        asyncio.create_task(self.flush_log())


GLOBAL_MASTER: Master = None


def get_master() -> Master:
    """获取全局的 Master 对象，用于跨节点共享数据和同步计数"""
    global GLOBAL_MASTER
    if GLOBAL_MASTER is None:
        GLOBAL_MASTER = ray.get_actor("Master")
    return GLOBAL_MASTER


def set(key: str, value: object):
    """将数据推送到全局的 Master 对象，比如config"""
    return get_master().push.remote(key, value)


def get(key: str) -> object:
    """从全局的 Master 对象获取数据"""
    return ray.get(get_master().get.remote(key))


def register(key: str) -> int:
    """注册一个全局的计数器，同步计数，比如 Actor 的 ID"""
    return ray.get(get_master().register.remote(key))

def log(msg: str, flush: bool=False):
    """输出日志到全局的 Master 对象，日志目录位于当前实验目录下"""
    return get_master().log.remote(msg, flush)


if __name__ == "__main__":
    ray.init(namespace="master")

    import os
    if not os.path.exists("master"):
        os.makedirs("master")

    master = ray.remote(Master).options(
        num_cpus=0, name="Master", get_if_exists=True
    ).remote(time_window=60)

    config = {"a": 1, "b": 2}
    ray.get(set("config", config))
    assert get("config") == config

    assert register("actor") == 0
    assert register("actor") == 1

    @ray.remote
    def test():
        assert get("config") == config
        assert register("actor") == 2
        ray.get(set("config", "hello"))

    config2 = {"a": 2, "b": 3}
    ray.get(test.remote())
    assert get("config") == "hello"
    assert register("actor") == 3
    ray.get(log("hello world 1", flush=True))
    ray.get(log("hello world 2", flush=True))
    ray.get(log("hello world 3", flush=True))
    ray.get(log("hello world 4", flush=True))
    print("Test success!")
