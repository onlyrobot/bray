import numpy as np
import torch
import ray

from typing import NewType

# 可以是单个 numpy/tensor 数组，也可以是一个 numpy/tensor 数组的列表，
# 或者是一个 numpy/tensor 数组的字典，或者是它们的嵌套
NestedArray = NewType("NestedArray", any)


def handle_nested_array(inputs: NestedArray, handler) -> NestedArray:
    if isinstance(inputs, (np.ndarray, torch.Tensor)):
        return handler(inputs)
    elif isinstance(inputs, (list, tuple)):
        return [handle_nested_array(i, handler) for i in inputs]
    elif isinstance(inputs, dict):
        return {k: handle_nested_array(v, handler) for k, v in inputs.items()}
    else:
        raise TypeError("Unsupported type: {}".format(type(inputs)))


@ray.remote
class TorchModelWorker:
    def __init__(self, model: torch.nn.Module):
        model.requires_grad_(False)
        model.eval()
        self.model = model

    async def forward(self, inputs: NestedArray) -> NestedArray:
        inputs = handle_nested_array(inputs, torch.as_tensor)
        inputs = handle_nested_array(inputs, lambda x: x.unsqueeze(0))
        outputs = self.model(inputs)
        return handle_nested_array(outputs, lambda x: x.squeeze(0).numpy())

    def set_weights(self, weights: NestedArray):
        self.model.set_weights(weights)


@ray.remote
class WeightsManager:
    def __init__(self, model_workers, weights: NestedArray):
        self.version = 0
        self.model_workers = model_workers
        self.weights = weights

    def set_weights(self, weights: NestedArray, version):
        for model_worker in self.model_workers:
            model_worker.set_weights.remote(weights)
        self.version = version


class RemoteModel:
    """
    RemoteModel封装了一个PyTorch模型，它会在Ray集群中创建多个TorchModelWorker实现并行计算
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: 目前支持PyTorch模型
        """
        self.model = model
        self.workers = [TorchModelWorker.remote(model) for _ in range(10)]
        initial_weights = model.get_weights()
        self.weights_manager = WeightsManager.remote(self.workers, initial_weights)
        self.worker_index = 0

    def forward(self, inputs: NestedArray) -> NestedArray:
        """
        执行模型的前向计算，返回模型的输出。
        Args:
            inputs: 模型的输入
        Returns:
            模型的输出
        """
        worker_index = self.worker_index % len(self.workers)
        self.worker_index += 1
        return ray.get(self.workers[worker_index].forward.remote(inputs))

    def get_model(self) -> torch.nn.Module:
        """
        获取被封装的原始模型，在Trainer里面会用到
        Returns:
            被封装的Pytorch模型
        """
        return self.model
    
    
    def publish_weights(self, weights: NestedArray, version: int):
        """
        发布模型的权重，会通知所有的ModelWorker更新权重
        Args:
            weights: 模型的权重，为一个numpy数组
            version: 权重的版本号，每次更新权重都需要增加版本号
        """
        self.weights_manager.set_weights.remote(weights, version)
