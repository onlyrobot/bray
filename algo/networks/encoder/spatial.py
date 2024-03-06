import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 4, 2, 1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 4, 2, 1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.layers = nn.Linear(128, embedding_dim)

    def forward(self, inputs: torch.Tensor):
        inputs = inputs.permute(0, 3, 1, 2)
        embedding = self.conv(inputs)
        embedding = self.layers(torch.flatten(embedding, start_dim=1))
        return embedding
