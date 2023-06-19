import ray


@ray.remote
class ModelWorker:
    def __init__(self, model) -> None:
        self.model = model

    async def forward(self, input):
        return self.model(input)

    def set_weights(self, weights, version):
        self.model.set_weights(weights, version)


@ray.remote
class ModelWeights:
    def __init__(self, model_workers, weights) -> None:
        self.version = 0
        self.model_workers = model_workers
        self.weights = weights

    def set_weights(self, weights, version):
        self.version = version
        for model_worker in self.model_workers:
            model_worker.set_weights.remote(weights, self.version)


class RemoteModel:
    def __init__(self, name: str, model) -> None:
        self.model = model
        self.workers = [ModelWorker.remote(model) for _ in range(10)]
        self.model_weights = ModelWeights.remote(self.workers, model.get_weights())

    def forward(self, input):
        return ray.get(self.workers[0].forward.remote(input))

    def publish_weights(self, weights, version):
        self.model_weights.set_weights.remote(weights, version)

    def get_model(self):
        return self.model
