import torch
import torch.nn.functional as F
import bray

def cal_kl(lhs_logit, rhs_logit):
    lhs_a = lhs_logit - torch.max(lhs_logit, dim=-1, keepdim=True).values
    rhs_a = rhs_logit - torch.max(rhs_logit, dim=-1, keepdim=True).values
    lhs_ea = torch.exp(lhs_a)
    rhs_ea = torch.exp(rhs_a)
    lhs_z = torch.sum(lhs_ea, dim=-1, keepdim=True)
    rhs_z = torch.sum(rhs_ea, dim=-1, keepdim=True)
    lhs_p = lhs_ea / lhs_z
    kl = lhs_p * (lhs_a - torch.log(lhs_z) - rhs_a + torch.log(rhs_z))
    return torch.sum(kl, dim=-1, keepdim=False)


def cal_policy(logit, behavior_logit, action, advantage):
    neglogp = F.cross_entropy(
        input=logit, target=action, reduction="none"
    )
    behavior_neglogp = F.cross_entropy(
        input=behavior_logit, target=action, reduction="none"
    )
    ratio = torch.exp(behavior_neglogp - neglogp)
    surr_loss1 = ratio * advantage
    surr_loss2 = torch.clip(ratio, 1.0 - 0.1, 1.0 + 0.1) * advantage

    return torch.min(surr_loss1, surr_loss2)


def cal_value(value, behavior_value, v_clip=10):
    value_loss1 = (value - behavior_value) ** 2
    vclipped = value + torch.clip(value - value, -v_clip, v_clip)
    value_loss2 = (vclipped - behavior_value) ** 2
    return torch.mean(torch.max(value_loss1, value_loss2))


def cal_action(
    name, logit, behavior_logit, action, advantage, valid_action
):
    loss = torch.tensor(0.0)
    valid_rate = valid_action.mean()
    bray.merge("action_valid_rate/" + name, valid_rate)
    if valid_rate == 0.0:
        return loss
    policy_loss = -torch.mean(
        cal_policy(
            logit,
            behavior_logit,
            action,
            advantage,
        )
        * valid_action
    )
    policy_loss /= valid_rate
    bray.merge("loss/policy_" + name, policy_loss)
    loss += policy_loss
    p = torch.softmax(logit, axis=-1)
    entropy_loss = p * torch.log(p + 1e-12)
    entropy_loss = torch.mean(
        entropy_loss.sum(axis=-1) * valid_action,
    )
    entropy_loss /= valid_rate
    bray.merge("loss/entropy_" + name, entropy_loss)
    loss += 0.01 * entropy_loss
    kl_loss = torch.mean(
        cal_kl(
            behavior_logit,
            logit,
        )
        * valid_action
    )
    kl_loss /= valid_rate
    bray.merge("loss/kl_" + name, kl_loss)
    loss += 0.5 * kl_loss


def loss(model: torch.nn.Module, replay: bray.NestedArray):
    obs, value, action, valid_action, advantage = (
        replay["obs"],
        replay["value"],
        replay["action"],
        replay["valid_action"],
        replay["advantage"],
    )
    target_value = advantage + value
    
    outputs = model(obs, action)
    value_loss = cal_value(outputs["value"], target_value)
    bray.merge("loss/value", value_loss)
    loss = 0.5 * value_loss
    for name in valid_action.keys():
        logit = name + "_logit"
        action_loss = cal_action(
            name,
            outputs[logit],
            action[logit],
            action[name],
            advantage,
            valid_action[name],
        )
        loss += action_loss
    bray.merge("loss/total", loss)
    return loss

