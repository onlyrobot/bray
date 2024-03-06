import tensorflow as tf
from ..layer import pooling
from tensorflow import keras


class Attention(keras.layers.Layer):
    def __init__(self, num_query, num_head, head_size):
        super(Attention, self).__init__()
        self._pooling = pooling.AttentionPooling(num_query, num_head, head_size)

    def call(self, inputs, mask, unit_length, name, tf_log_dict):
        return self._pooling(inputs, mask, unit_length, name, tf_log_dict)


class Max:
    def __init__(self, axis=1):
        self._axis = axis

    def __call__(self, inputs, *unused):
        outputs = tf.reduce_max(inputs, axis=self._axis)
        return outputs
