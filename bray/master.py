import ray


@ray.remote
class Master:
    def __init__(self, model, actor, config) -> None:
        pass

    def set_model_weights(self, weights, version):
        return version + 1
    
def run():
    ray.init()
    model = None
    actor = None
    config = None
    master = Master.remote(model, actor, config)
    version = 0
    for _ in range(10):
        version = ray.get(master.set_model_weights.remote(None, version))
    print(version)