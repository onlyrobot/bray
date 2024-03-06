import tensorflow as tf


def filter_by_prefix(inputs_dict, prefix):
    res_dict = {}
    for key, value in inputs_dict.items():
        if key.startswith(prefix):
            res_dict[key] = value
    return res_dict

def calculate_negative_logp(distribution_dict, logits_dict, action_dict, valid_action_dict):
    neg_logp_dict = {}
    for action_name, distribution_fn in distribution_dict.items():
        distribution = distribution_fn(logits_dict["logits_" + action_name])
        neg_logp_dict[action_name] = distribution.negative_logp(action_dict["action_" + action_name])
        neg_logp_dict[action_name] *= valid_action_dict["valid_action_"+action_name]
    return neg_logp_dict


def calculate_entropy(distribution_dict, logits_dict, valid_action_dict):
    entropy_dict = {}
    entropy_log_dict = {}
    max_logit_dict = {}
    min_logit_dict = {}
    for action_name, distribution_fn in distribution_dict.items():
        distribution = distribution_fn(logits_dict["logits_" + action_name])
        entropy_dict[action_name] = distribution.entropy() * valid_action_dict["valid_action_"+action_name]
        entropy_log_dict[action_name + "_entropy"] = tf.reduce_sum(
            distribution.tf_log_entropy() * valid_action_dict["valid_action_"+action_name]
        ) / tf.reduce_sum(valid_action_dict["valid_action_"+action_name])
        max_logit_dict[action_name + "_max_logit"] = tf.clip_by_value(
            tf.reduce_max(tf.reduce_max(distribution.logits, axis=-1)), 0, 1e10
        )
        log_min_logits = tf.where(distribution.logits < -1e5, tf.ones_like(distribution.logits), distribution.logits)
        min_logit_dict[action_name + "_min_logit"] = tf.reduce_min(tf.reduce_min(log_min_logits, axis=-1))

    return entropy_dict, entropy_log_dict, max_logit_dict, min_logit_dict

def calculate_kl(distribution_dict, logits_dict, other_logits_dict, valid_action_dict):
    kl_dict = {}
    for action_name, distribution_fn in distribution_dict.items():
        distribution = distribution_fn(logits_dict["logits_" + action_name])
        other_distribution = distribution_fn(other_logits_dict["logits_" + action_name])
        kl_dict[action_name] = distribution.kl(other_distribution) * valid_action_dict["valid_action_"+action_name]
    return kl_dict


class PPOLoss:
    def __init__(self, clip_param=0.1, vf_clip_param=2, vf_loss_coef=1, entropy_coef=0.01):
        self._clip_param = clip_param
        self._vf_clip_param = vf_clip_param
        self._vf_loss_coef = vf_loss_coef
        self._entropy_coef = entropy_coef

    def __call__(self, old_neg_logp, neg_logp, advantage, old_value, value, target_value, entropy, multi_head_value_config):
        ratio = tf.exp(old_neg_logp - neg_logp)
        surr1 = ratio * advantage
        surr2 = tf.clip_by_value(ratio, 1 - self._clip_param, 1 + self._clip_param) * advantage
        surr = tf.minimum(surr1, surr2)
        policy_loss = -tf.reduce_mean(surr)
        value_loss_dict = dict()

        value_loss = 0
        for i in range(multi_head_value_config["num_heads"]):

            value_pred_clip = old_value[:, i] + tf.clip_by_value(
                value[:, i] - old_value[:, i], -self._vf_clip_param, self._vf_clip_param
            )
            value_loss1 = tf.square(value[:, i] - target_value[:, i])
            value_loss2 = tf.square(value_pred_clip - target_value[:, i])
            single_value_head_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))
            value_loss_dict[multi_head_value_config["value_loss_heads"][i] + "_value_loss"] = single_value_head_loss
            value_loss += single_value_head_loss

        loss = policy_loss + value_loss * self._vf_loss_coef - entropy * self._entropy_coef
        clip_prob = tf.reduce_mean(tf.where(surr1 > surr2, tf.ones_like(surr1), tf.zeros_like(surr1)))
        ratio_diff = tf.reduce_mean(tf.abs(ratio - 1.0))
        return loss, policy_loss, value_loss, clip_prob, ratio_diff, value_loss_dict
