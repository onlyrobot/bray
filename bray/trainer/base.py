import torch
from bray.utils.nested_array import NestedArray


class Trainer:
    """
    Trainer是一个用于训练模型的类，包含了训练数据处理、Loss计算、Eval评估等功能，
    实现该类的时候需要保证分布式拓展能力，如必要可以直接使用horovod相关API
    """
    def __init__(
            self, name: str, config: dict, model: torch.nn.Module, optimizer
        ):
        """
        从指定模型构造一个Trainer，这里可以将模型保存下来，
        但不要修改模型的设备、数据类型等参数，以防止外部无法识别模型
        Args:
            name: Trainer名称，在配置文件中指定
            config: 全局配置，通过 `config[name]` 获取当前Trainer的配置
            model: 一个torch.nn.Module
            optimizer: 包装模型的优化器，可以动态修改学习率等优化器参数
        """
        raise NotImplementedError

    def handle(self, replay: NestedArray) -> NestedArray:
        """
        这里的replay来自Buffer的pop，这里可以定义对replay的预处理逻辑
        Args:
            replay: 处理前的Replay数据，是一个NestedArray
        Returns:
            处理后的Replay数据，是一个NestedArray，要求设备位置保持一致
        """
        return replay

    def loss(self, replay: NestedArray, step: int) -> torch.Tensor:
        """
        计算Loss，计算过程中的指标可以用bray的相关API输出
        Args:
            replay: 一个Replay，是一个NestedArray
            step: 当前的样本数
        Returns:
            Loss，是一个torch.Tensor，保证可以反向传播
        """
        raise NotImplementedError
    
    def eval(self, replay: NestedArray):
        """
        评估模型，这里可以用bray的相关API输出评估指标，但不要修改模型的设备、权重等信息
        Args:
            replay: 一个Replay，是一个NestedArray，通常来自Eval Buffer
        """
        raise NotImplementedError
