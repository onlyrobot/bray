import torch
import bray
import horovod.torch as hvd
from typing import Iterator


def cal_kl(lhs_logits, rhs_logits):
    lhs_a = lhs_logits - torch.max(lhs_logits, dim=-1, keepdim=True).values
    rhs_a = rhs_logits - torch.max(rhs_logits, dim=-1, keepdim=True).values
    lhs_ea = torch.exp(lhs_a)
    rhs_ea = torch.exp(rhs_a)
    lhs_z = torch.sum(lhs_ea, dim=-1, keepdim=True)
    rhs_z = torch.sum(rhs_ea, dim=-1, keepdim=True)
    lhs_p = lhs_ea / lhs_z
    kl = lhs_p * (lhs_a - torch.log(lhs_z) - rhs_a + torch.log(rhs_z))
    return torch.sum(kl, dim=-1, keepdim=False)


def cal_entroy(logits):
    a = logits - torch.max(logits, axis=-1, keepdim=True).values
    ea = torch.exp(a)
    z = torch.sum(ea, dim=-1, keepdim=True)
    p = ea / z
    return torch.sum(p * (torch.log(z) - a), axis=-1, keepdim=False)


class AtariTrainer(bray.Trainer):
    def __init__(self, config):
        self.config = config
        hvd.init()

    def train_step(self, model, batch_replay, optimizer):
        obs, value, logit, action, advantage = (
            torch.as_tensor(batch_replay["obs"]),
            torch.as_tensor(batch_replay["value"]),
            torch.as_tensor(batch_replay["logit"]),
            torch.as_tensor(batch_replay["action"]),
            torch.as_tensor(batch_replay["advantage"]),
        )
        local_batch_size = advantage.shape[0]
        global_batch_size = local_batch_size * hvd.size()
        target_value = advantage + value

        advantage_sum = torch.sum(advantage)
        advantage_sum = hvd.allreduce(advantage_sum, op=hvd.Sum)
        advantage_mean = advantage_sum / global_batch_size
        squared_error = torch.sum((advantage - advantage_mean) ** 2)
        squared_error_sum = hvd.allreduce(squared_error, op=hvd.Sum)
        advantages_variance = squared_error_sum / global_batch_size
        advantage = (advantage - advantage_mean) / torch.sqrt(
            advantages_variance + 1e-8
        )

        optimizer.zero_grad()
        t_value, t_logit, _ = model(obs)
        target_neglogp = torch.nn.functional.cross_entropy(
            input=t_logit, target=action, reduction="none"
        )
        neglogp = torch.nn.functional.cross_entropy(
            input=logit, target=action, reduction="none"
        )
        ratio = torch.exp(neglogp - target_neglogp)
        surr_loss1 = ratio * advantage
        surr_loss2 = torch.clip(ratio, 1.0 - 0.1, 1.0 + 0.1) * advantage

        policy_loss = torch.min(surr_loss1, surr_loss2)
        policy_loss = -torch.mean(policy_loss, dim=0, keepdim=False)

        value_loss1 = (t_value - target_value) ** 2
        vclipped = value + torch.clip(t_value - value, -10, 10)

        value_loss2 = (vclipped - target_value) ** 2
        value_loss = torch.max(value_loss1, value_loss2)
        value_loss = 0.5 * torch.mean(value_loss, dim=0, keepdim=False)

        entropy_loss = torch.mean(cal_entroy(t_logit), dim=0, keepdim=False)

        kl_loss = torch.mean(cal_kl(logit, t_logit), dim=0, keepdim=False)

        loss = policy_loss + value_loss - 0.01 * entropy_loss + 0.5 * kl_loss
        loss.backward()
        optimizer.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        with optimizer.skip_synchronize():
            optimizer.step()
        bray.merge("loss", loss)
        bray.merge("policy_loss", policy_loss)
        bray.merge("value_loss", value_loss)
        bray.merge("entropy_loss", entropy_loss)
        bray.merge("kl_loss", kl_loss)

    def train(
        self,
        remote_model: bray.RemoteModel,
        buffer: Iterator[bray.NestedArray],
        num_steps: int,
    ):
        # initialize model
        model = remote_model.get_model()
        weights, step = remote_model.get_weights()
        with torch.no_grad():
            bray.set_torch_model_weights(model, weights)
        # initialize optimizer
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=5e-4)
        optimizer = hvd.DistributedOptimizer(optimizer)
        # initialize buffer
        buffer = bray.BatchBuffer(buffer, batch_size=8)
        for i in range(num_steps):
            self.train_step(model, next(buffer), optimizer)
            weights = bray.get_torch_model_weights(model)
            if hvd.rank() == 0 and i % 4 == 0:
                remote_model.publish_weights(weights, step + i)
            print(f"Train global step {step + i}, local step {i}")
        print(f"Train {num_steps} steps done!")
