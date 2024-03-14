import tensorflow as tf
from .attention import MultiHeadAttention
from tensorflow import keras


class AttentionPooling(keras.layers.Layer):
    def __init__(self, num_query, num_head, head_size):
        super(AttentionPooling, self).__init__()
        self._multi_head_attention = MultiHeadAttention(num_head, head_size)
        initializer = keras.initializers.RandomNormal()
        self._query = self.add_weight(
            name="attention_pooling_query",
            shape=(num_query, num_head * head_size),
            initializer=initializer,
            trainable=True,
        )

    def call(self, inputs, mask, unit_length, name, tf_log_dict):
        query = tf.tile(tf.expand_dims(self._query, axis=0), [tf.shape(inputs)[0], 1, 1])
        outputs = self._multi_head_attention(
            query, inputs, inputs, v_mask=mask, unit_length=unit_length, name=name, tf_log_dict=tf_log_dict
        )
        outputs = keras.layers.Flatten()(outputs)
        return outputs
