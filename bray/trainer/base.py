from typing import Iterator

from bray.model.model import RemoteModel
from bray.utils.nested_array import NestedArray


class Trainer:
    """
    这里定义了一个抽象的 Trainer 类，用于训练模型，用户需要继承这个类并实现 train 函数，
    然后将这个类传给 RemoteTrainer，RemoteTrainer 会在多个节点上创建 Trainer 的实例
    """

    def __init__(self, config: any):
        """
        在这里初始化 Trainer，可以从 config 中读取配置

        Args:
            config: Trainer 的配置，来自 RemoteTrainer 的构造函数参数
        """
        raise NotImplementedError

    def train(self, remote_model: RemoteModel, replays: Iterator[NestedArray]):
        """
        在指定的RemoteBuffer上训练模型，这个函数会被 RemoteTrainer 调用

        ``` python
        remote_trainer.train(remote_model, remote_buffer)
        ```

        Args:
            remote_model: 需要训练的模型，调用 `remote_model.get_model()` 可以获取原始模型，
                也就是用户传给 RemoteModel 的那个模型，它是一个 torch.nn.Module 的实例，
                训练好的权重需要调用 `remote_model.publish_weights(weights, version)` 发布
            replays: 从 RemoteBuffer 中读取的数据，是一个生成器，每次调用 next() 会返回一个 NestedArray
        """
        raise NotImplementedError
