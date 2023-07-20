import torch
import bray
import horovod.torch as hvd
import time


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


def train_step(
    remote_model,
    replay,
    model,
    optimizer,
    weights_publish_interval,
    step,
):
    obs, value, logit, action, advantage = (
        replay["obs"],
        replay["value"],
        replay["logit"],
        replay["action"],
        replay["advantage"],
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
    advantage = (advantage - advantage_mean) / torch.sqrt(advantages_variance + 1e-8)

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
    if hvd.rank() != 0:
        return
    bray.merge("loss", policy_loss, type="policy")
    bray.merge("loss", value_loss, type="value")
    bray.merge("loss", entropy_loss, type="entropy")
    bray.merge("loss", kl_loss, type="kl")
    bray.merge(
        "loss",
        loss,
        desc={"time_window_avg": "total = policy + value - 0.01 * entropy + 0.5 * kl"},
    )
    if step % weights_publish_interval == 0:
        weights = bray.get_torch_model_weights(model)
        remote_model.publish_weights(weights)
    if step % 100 == 0:
        print(f"Train step {step}, loss: {loss.item()}")


def train_atari(
    model,
    buffer,
    batch_size,
    weights_publish_interval,
    num_steps,
):
    to_gpu = torch.cuda.is_available()
    # initialize model
    remote_model = bray.RemoteModel(name=model)
    model = remote_model.get_model()
    if to_gpu:
        model.cuda()
    # initialize optimizer
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    hvd.init()
    optimizer = hvd.DistributedOptimizer(optimizer)
    # initialize buffer
    remote_buffer = bray.RemoteBuffer(name=buffer)
    buffer = bray.BatchBuffer(remote_buffer, batch_size=batch_size)
    buffer = bray.TorchTensorBuffer(buffer, to_gpu=to_gpu)
    buffer = bray.PrefetchBuffer(buffer, max_reuse=8, name=remote_buffer.name)
    for i in range(num_steps):
        beg = time.time()
        replay = next(buffer)
        end = time.time()
        bray.merge(
            "replay",
            (end - beg) * 1000,
            desc={"time_window_avg": "next replay latency ms"},
        )

        train_step(
            remote_model,
            replay,
            model,
            optimizer,
            weights_publish_interval,
            step=i,
        )
        if hvd.rank() != 0:
            continue
        bray.merge(
            "train",
            (time.time() - end) * 1000,
            desc={
                "time_window_avg": "train step latency ms",
                "time_window_cnt": "train step per minute",
            },
        )
    print(f"Train all {num_steps} steps done!")
