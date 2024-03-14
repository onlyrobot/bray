import tensorflow as tf

RNG = tf.random.get_global_generator()


def tf_log_softmax_entropy_with_logits(logits):
    """根据logits，计算softmax后的entropy
    为了避免overflow和underflow的问题，计算softmax的方式会有一点变化，
        参考 https://www.deeplearningbook.org/contents/numerical.html
    Args:
        logits: [B, ..., num_classes]
    Returns:
        entropy: [B]
    """
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    # entropy的计算参考 https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L193
    entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
    # normalized entropy
    no_mask_action_length = tf.reduce_sum(tf.cast(logits > -1e5, dtype=tf.float32), axis=-1)
    entropy = entropy / tf.math.log(tf.cast(no_mask_action_length, dtype=entropy.dtype) + 1e-6)
    entropy = tf.where(no_mask_action_length > 1.0, entropy, tf.ones_like(entropy))
    return entropy


def softmax_entropy_with_logits(logits):
    """根据logits，计算softmax后的entropy
    为了避免overflow和underflow的问题，计算softmax的方式会有一点变化，
        参考 https://www.deeplearningbook.org/contents/numerical.html
    Args:
        logits: [B, ..., num_classes]
    Returns:
        entropy: [B]
    """
    a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    p0 = ea0 / z0
    # entropy的计算参考 https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L193
    entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
    # normalized entropy
    entropy = entropy / tf.math.log(tf.cast(tf.shape(logits)[-1], dtype=entropy.dtype) + 1e-6)
    return entropy


def softmax_kl_with_logits(logits, other_logits, temperature=1.0):
    """calculate kl of two distributions represented with softmax logits
    参考 https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L184
    logits: [B, ..., num_classes]
    other_logits: [B, ..., num_classes]
    """
    a0 = (logits - tf.reduce_max(logits, axis=-1, keepdims=True)) / tf.constant(temperature)
    a1 = (other_logits - tf.reduce_max(other_logits, axis=-1, keepdims=True)) / tf.constant(temperature)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    kl = tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)
    return kl


def sample_softmax_with_logits(logits, p_min=0.0, temp=1.0):
    """从softmax的logits表示的分布上采样"""
    if p_min > 0.0:
        probs = tf.nn.softmax(logits)
        logit_mask = tf.cast(probs < p_min, dtype=logits.dtype) * 1e5
        logits = logits - logit_mask
    u = RNG.uniform(tf.shape(logits))
    action = tf.argmax(logits / temp - tf.math.log(-tf.math.log(u)), axis=-1, output_type=tf.int32)
    return action
