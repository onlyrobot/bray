from bray.model.model import RemoteModel
from bray.buffer.buffer import Buffer


class Trainer:
    def __init__(self):
        raise NotImplementedError

    def train(self, remote_model: RemoteModel, replays: list[Buffer]):
        raise NotImplementedError
