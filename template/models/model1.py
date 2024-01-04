import numpy as np
import torch


class Model1(torch.nn.Module):
    def __init__(self, action_space=9):
        super().__init__()
        self.base_net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="zeros",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=2,
                padding_mode="zeros",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=256,
                kernel_size=11,
                stride=1,
                padding=0,
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.values_net = torch.nn.Linear(
            in_features=256,
            out_features=1,
        )
        self.logits_net = torch.nn.Linear(
            in_features=256,
            out_features=action_space,
        )

    def forward(self, state):
        images = state["image"]  # shape is [None, 4, 42, 42]
        logits = self.base_net(images)
        values = torch.squeeze(
            self.values_net(logits),
            dim=1,
        )
        logits = self.logits_net(logits)
        probs = torch.softmax(logits, dim=1)
        actions = torch.multinomial(
            probs,
            num_samples=1,
        )
        actions = torch.squeeze(actions, dim=1)
        return values, logits, actions


def build_model(name, config: dict) -> tuple[torch.nn.Module, tuple]:
    """
    构建PyTorch模型，用于推理和训练，返回模型和输入数据的样例
    Args:
        name: 模型名称，在配置文件中指定
        config: 全局配置，通过 `config[name]` 获取当前模型配置
    Returns:
        model: PyTorch模型
        forward_args: 保证 model(*forward_args) 可以正常执行
    """
    model = Model1()

    forward_args = ({"image": np.random.randn(1, 4, 42, 42).astype(np.float32)},)
    return model, forward_args
