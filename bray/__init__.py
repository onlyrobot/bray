import dataclasses
from bray.buffer.buffer import Buffer, RemoteBuffer
from bray.model.model import RemoteModel
from bray.trainer.trainer import RemoteTrainer
from bray.actor.actor import RemoteActor


@dataclasses.dataclass
class Agent:
    """
    Agent代表了Actor中的一个智能体，这里定义了它的基本属性
    """

    remote_model: RemoteModel
    remote_buffer: RemoteBuffer


class Actor:
    """
    Actor是一个有状态服务接受来自Gamecore的step调用，调用的顺序是：
    start(__init__) -> tick -> tick -> ... -> end
    """

    def __init__(
        self, agents: dict[str, Agent], config: any, game_id: str, data: bytes
    ):
        """
        初始化一个新的Actor，当一局新的游戏开始时，会调用这个方法
        Args:
            agents: 一个字典，key是agent的名字，value是agent的实例
            config: 一个任意的配置对象，由RemoteActor传入
            game_id: 一个唯一的游戏ID，由Gamecore传入
            data: 一个任意的字节串，由Gamecore传入，通常是空的
        """
        raise NotImplementedError

    def tick(self, round_id: int, data: bytes) -> bytes:
        """
        执行一步游戏，由Gamecore调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态
        2. 调用agent的remote_model.forward方法，获取action
        3. 将action序列化为字节串，返回给Gamecore
        4. 收集trajectory，将其push到agent的remote_buffer中
        Args:
            round_id: 当前游戏的回合数
            data: 一个任意的字节串，由Gamecore传入，通常是游戏状态
        Returns:
            一个任意的字节串，通常是序列化后的action
        """
        raise NotImplementedError

    def end(self, round_id: int, data: bytes) -> bytes:
        """
        游戏结束时调用，由Gamecore调用，在这里需要执行以下操作：
        1. 从data中解析出游戏状态（通常是最后一帧的reward）
        2. 终止收集trajectory，将其push到agent的remote_buffer中
        3. 进行一些清理工作
        Args:
            round_id: 当前游戏的回合数
            data: 一个任意的字节串，由Gamecore传入
        Returns:
            一个任意的字节串，通常是空的，或者一些统计信息
        """
        raise NotImplementedError


class Trainer:
    def __init__(self):
        raise NotImplementedError

    def train(self, remote_model: RemoteModel, replays: list[Buffer]):
        raise NotImplementedError
