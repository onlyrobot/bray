import torch


class AtariModel(torch.nn.Module):
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

    def forward(self, images):
        # transpose [None, 42, 42, 4] into [None, 4, 42, 42]
        images = torch.permute(images, (0, 3, 1, 2))
        logits = self.base_net(images)
        values = torch.squeeze(
            self.values_net(logits),
            dim=1,
        )
        logits = self.logits_net(logits)
        actions = torch.multinomial(
            torch.exp(logits),
            num_samples=1,
        )
        actions = torch.squeeze(actions, dim=1)
        return values, logits, actions
