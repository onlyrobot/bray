from bray.buffer.buffer import Buffer, RemoteBuffer
from bray.model.model import RemoteModel
from bray.trainer.trainer import RemoteTrainer
from bray.actor.actor import RemoteActor
from bray.agent.agent import Agent


class Actor:
    def __init__(
        self, agents: dict[str, Agent], config: any, game_id: str, data: bytes
    ):
        raise NotImplementedError

    def tick(self, round_id: int, data: bytes) -> bytes:
        raise NotImplementedError

    def end(self, round_id: int, data: bytes) -> bytes:
        raise NotImplementedError


class Trainer:
    def __init__(self):
        raise NotImplementedError

    def train(self, remote_model: RemoteModel, replays: list[Buffer]):
        raise NotImplementedError