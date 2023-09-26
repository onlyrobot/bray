import bray
import tensorflow as tf
import horovod.tensorflow as hvd
import time


def cal_kl(lhs_logits, rhs_logits):
    lhs_a = lhs_logits - tf.reduce_max(lhs_logits, axis=-1, keepdims=True)
    rhs_a = rhs_logits - tf.reduce_max(rhs_logits, axis=-1, keepdims=True)
    lhs_ea = tf.exp(lhs_a)
    rhs_ea = tf.exp(rhs_a)
    lhs_z = tf.reduce_sum(lhs_ea, axis=-1, keepdims=True)
    rhs_z = tf.reduce_sum(rhs_ea, axis=-1, keepdims=True)
    lhs_p = lhs_ea / lhs_z
    kl = lhs_p * (lhs_a - tf.math.log(lhs_z) - rhs_a + tf.math.log(rhs_z))
    return tf.reduce_sum(kl, axis=-1, keepdims=False)


@tf.function
def train_step_(
    obs,
    value,
    logit,
    action,
    advantage,
    model: tf.keras.Model,
    optimizer: tf.optimizers.Optimizer,
):
    local_batch_size = advantage.shape[0]
    global_batch_size = local_batch_size * hvd.size()
    target_value = value + advantage

    advantage_sum = tf.reduce_sum(advantage)
    advantage_sum = hvd.allreduce(advantage_sum, op=hvd.Sum)
    advantage_mean = advantage_sum / global_batch_size
    squared_error = tf.reduce_sum((advantage - advantage_mean) ** 2)
    squared_error_sum = hvd.allreduce(squared_error, op=hvd.Sum)
    advantages_variance = squared_error_sum / global_batch_size
    advantage = (advantage - advantage_mean) / tf.sqrt(advantages_variance + 1e-8)

    with tf.GradientTape() as tape:
        t_value, t_logit, _ = model(obs, training=True)
        target_neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=action, logits=t_logit
        )
        neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=action, logits=logit
        )
        ratio = tf.exp(neglogp - target_neglogp)
        surr_loss1 = ratio * advantage
        surr_loss2 = tf.clip_by_value(ratio, 1.0 - 0.1, 1.0 + 0.1) * advantage

        policy_loss = tf.minimum(surr_loss1, surr_loss2)
        policy_loss = -tf.reduce_mean(policy_loss, axis=0, keepdims=False)

        value_loss1 = tf.square(t_value - target_value)
        vclipped = value + tf.clip_by_value(t_value - value, -10, 10)

        value_loss2 = tf.square(vclipped - target_value)
        value_loss = tf.maximum(value_loss1, value_loss2)
        value_loss = 0.5 * tf.reduce_mean(value_loss, axis=0, keepdims=False)

        exp_logit = tf.exp(t_logit)
        prob = exp_logit / tf.reduce_sum(exp_logit, axis=-1, keepdims=True)
        entropy_loss = -tf.reduce_sum(tf.math.log(prob + 1e-6) * prob, axis=-1)
        entropy_loss = tf.reduce_mean(entropy_loss, axis=0, keepdims=False)

        kl_loss = tf.reduce_mean(cal_kl(logit, t_logit), axis=0, keepdims=False)

        loss = policy_loss + value_loss - 0.01 * entropy_loss + 0.5 * kl_loss
    tape = hvd.DistributedGradientTape(tape)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return policy_loss, value_loss, entropy_loss, kl_loss, loss


def train_step(
    remote_model,
    replay,
    model: tf.keras.Model,
    optimizer: tf.optimizers.Optimizer,
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
    policy_loss, value_loss, entropy_loss, kl_loss, loss = train_step_(
        obs, value, logit, action, advantage, model, optimizer
    )
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
        weights = model.get_weights()
        remote_model.publish_weights(weights)
    if step % 100 == 0:
        print(f"Train step {step}, loss: {loss}")


def train_atari(
    remote_model,
    remote_buffer,
    batch_size,
    weights_publish_interval,
    num_steps,
):
    model = remote_model.get_model()
    # initialize model
    hvd.init()
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    if gpus := tf.config.list_physical_devices("GPU"):
        tf.config.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    # initialize optimizer
    optimizer = tf.optimizers.Adam(5e-4)
    hvd.broadcast_variables(model.variables, root_rank=0)
    hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    # initialize buffer
    buffer = bray.BatchBuffer(remote_buffer, batch_size=batch_size)
    buffer = bray.TensorFlowTensorBuffer(buffer)
    buffer = bray.PrefetchBuffer(buffer, max_reuse=4, name=remote_buffer.name)
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
