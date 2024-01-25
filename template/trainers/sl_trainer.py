import torch
import bray


class Trainer1(bray.Trainer):
    def __init__(self, name, config: dict, model: torch.nn.Module):
        self.model = model

    def handle(self, replay: bray.NestedArray) -> bray.NestedArray:
        return replay

    def loss(self, replay: bray.NestedArray) -> torch.Tensor:
        state, action = replay["state"], replay["action"]

        _, logit, _ = self.model(state)
        loss = torch.nn.functional.cross_entropy(
            input=logit, target=action, reduction="none"
        )
        loss = loss.mean()
        bray.merge("loss/total", loss)
        return loss

    def eval(self, replay: bray.NestedArray):
        state, action = replay["state"], replay["action"]
        _, logit, _ = self.model(state)
        loss = torch.nn.functional.cross_entropy(
            input=logit, target=action, reduction="none"
        )
        loss = loss.mean()
        bray.merge("eval/loss/total", loss)
