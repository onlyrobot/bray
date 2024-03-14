from __future__ import annotations

import tensorflow as tf
from .utils import softmax_entropy_with_logits
from .utils import softmax_kl_with_logits
from .utils import tf_log_softmax_entropy_with_logits


class Categorical:
    def __init__(self, logits, temperature=1.0):
        self._logits = logits
        self._temperature = temperature

    @property
    def logits(self):
        return self._logits

    def sample(self, seed=None):
        u = tf.random.uniform(tf.shape(self._logits), minval=1e-6, seed=seed)
        action = tf.argmax(
            self._logits / self._temperature - tf.math.log(-tf.math.log(u)), axis=-1, output_type=tf.int32
        )
        action = tf.stop_gradient(action)
        return action

    def negative_logp(self, action):
        action = tf.cast(action, dtype=tf.int32)
        neg_logp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=self._logits)
        return neg_logp

    def entropy(self):
        return softmax_entropy_with_logits(self._logits)

    def tf_log_entropy(self):
        return tf_log_softmax_entropy_with_logits(self._logits)

    def kl(self, other: Categorical):
        return softmax_kl_with_logits(self._logits, other.logits)
