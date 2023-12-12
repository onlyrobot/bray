import ray


@ray.remote(num_cpus=0, name="Master")
class Master:
    def __init__(self):
        self.registery, self.data = {}, {}

    def push(self, key: str, value: object):
        self.data[key] = value

    def get(self, key: str) -> object:
        return self.data[key]

    def register(self, key: str) -> int:
        id = self.registery.get(key, 0)
        self.registery[key] = id + 1
        return id


GLOBAL_MASTER = None


def get_master() -> Master:
    """获取全局的 Master 对象，用于便捷地跨节点共享数据和同步计数"""
    global GLOBAL_MASTER
    if GLOBAL_MASTER is None:
        GLOBAL_MASTER = Master.remote()
    return GLOBAL_MASTER


def push(key: str, value: object):
    """将数据推送到全局的 Master 对象，支持任意可序列化的数据，比如config"""
    get_master().push.remote(key, value)


def get(key: str) -> object:
    """从全局的 Master 对象获取数据，本调用为堵塞调用，所以不要频繁调用"""
    return ray.get(get_master().get.remote(key))


def register(key: str) -> int:
    """注册一个全局的计数器，用于在多个节点之间同步计数，比如 Actor 的 ID"""
    return ray.get(get_master().register.remote(key))


if __name__ == "__main__":
    ray.init()
    
    config = {"a": 1, "b": 2}
    push("config", config)
    assert get("config") == config

    assert register("actor") == 0
    assert register("actor") == 1

    @ray.remote
    def test():
        assert get("config") == config
        assert register("actor") == 2
        push("config", "hello")

    config2 = {"a": 2, "b": 3}
    ray.get(test.remote())
    assert get("config") == "hello"
    assert register("actor") == 3
    print("Test success!")