import torch
import bray


def cal_kl(lhs_logits, rhs_logits):
    lhs_a = lhs_logits - torch.max(lhs_logits, dim=-1, keepdim=True).values
    rhs_a = rhs_logits - torch.max(rhs_logits, dim=-1, keepdim=True).values
    lhs_ea = torch.exp(lhs_a)
    rhs_ea = torch.exp(rhs_a)
    lhs_z = torch.sum(lhs_ea, dim=-1, keepdim=True)
    rhs_z = torch.sum(rhs_ea, dim=-1, keepdim=True)
    lhs_p = lhs_ea / lhs_z
    return torch.sum(
        lhs_p * (lhs_a - torch.log(lhs_z) - rhs_a + torch.log(rhs_z)),
        dim=-1,
        keepdim=False,
    )


def cal_entroy(logits):
    a = logits - torch.max(logits, axis=-1, keepdim=True).values
    ea = torch.exp(a)
    z = torch.sum(ea, dim=-1, keepdim=True)
    p = ea / z
    return torch.sum(p * (torch.log(z) - a), axis=-1, keepdim=False)


class AtariTrainer(bray.Trainer):
    def __init__(self, config):
        self.config = config


    def train_step(self, model, replay):
        pass

    def train(self, remote_model, replays):
        model = remote_model.get_model()
        version = 1
        for replay in replays:
            self.train_step(model, replay)
            weights = model.get_weights()
            remote_model.publish_weights(weights, version)
            version += 1
            import time 
            time.sleep(1)
