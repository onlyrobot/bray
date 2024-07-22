import torch
import ray, bray


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


class AtariTrainer(bray.Trainer):
    def __init__(self, name, config, model, optimizer):
        self.config = config
        self.model, self.optimizer = model, optimizer
        self.world_size = ray.train.get_context().get_world_size()

    def loss(self, replay, step):
        state, value, logit, action, advantage = (
            replay["state"],
            replay["value"],
            replay["logit"],
            replay["action"],
            replay["advantage"],
        )
        local_batch_size = advantage.shape[0]
        # global_batch_size = local_batch_size * self.world_size
        global_batch_size = local_batch_size
        target_value = value + advantage

        advantage_sum = torch.sum(advantage)
        advantage_mean = advantage_sum / global_batch_size
        squared_error = torch.sum((advantage - advantage_mean) ** 2)
        advantages_variance = squared_error / global_batch_size
        advantage = (advantage - advantage_mean) / torch.sqrt(advantages_variance + 1e-8)

        t_value, t_logit, _ = self.model(state)
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

        exp_logit = torch.exp(t_logit)
        prob = exp_logit / torch.sum(exp_logit, dim=-1, keepdim=True)
        entropy_loss = -torch.sum(torch.log(prob + 1e-6) * prob, dim=-1)
        entropy_loss = torch.mean(entropy_loss, dim=0, keepdim=False)

        kl_loss = torch.mean(cal_kl(logit, t_logit), dim=0, keepdim=False)

        loss = policy_loss + value_loss - 0.01 * entropy_loss + 0.5 * kl_loss
        bray.merge("loss", policy_loss, type="policy")
        bray.merge("loss", value_loss, type="value")
        bray.merge("loss", entropy_loss, type="entropy")
        bray.merge("loss", kl_loss, type="kl")
        bray.merge("loss", loss, desc={
            "time_window_avg": "total = policy + value - 0.01 * entropy + 0.5 * kl"},
        )
        return loss