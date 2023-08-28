import torch


class GridShootingModel(torch.nn.Module):
    def __init__(self, action_space=9):
        super().__init__()
        self.base_net = torch.nn.Linear(
            in_features=56,
            out_features=16,
        )
        self.values_net = torch.nn.Linear(
            in_features=16,
            out_features=1,
        )
        self.logits_net = torch.nn.Linear(
            in_features=16,
            out_features=action_space,
        )

    def forward(self, state):
        obs, action_mask = state["obs"], state["action_mask"]
        logits = self.base_net(obs)
        values = torch.squeeze(
            self.values_net(logits),
            dim=1,
        )
        logits = self.logits_net(logits)
        logits = logits - (1 - action_mask) * 1e5
        actions = torch.multinomial(
            torch.exp(logits),
            num_samples=1,
        )
        actions = torch.squeeze(actions, dim=1)
        return values, logits, actions
