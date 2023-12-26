import torch
import bray
from rl_template.trainers.trainer1 import Trainer1 as RLTrainer
from sl_template.trainers.trainer1 import Trainer1 as SLTrainer


class Trainer1(bray.Trainer):
    def __init__(self, model: torch.nn.Module):
        self.rl_trainer = RLTrainer(model)
        self.sl_trainer = SLTrainer(model)

    def replay_handler(self, replay: bray.NestedArray) -> bray.NestedArray:
        return replay

    def loss(self, replay: bray.NestedArray) -> torch.Tensor:
        if "advantage" in replay:
            return self.rl_trainer.loss(replay)
        return self.sl_trainer.loss(replay)

    def eval(self, replay: bray.NestedArray):
        if "advantage" in replay:
            return
        return self.sl_trainer.eval(replay)
