import tensorflow as tf
from ..distribution.utils import tf_log_softmax_entropy_with_logits
from ..utils import apply_mask
from tensorflow import keras

class PatchRelativeMultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, num_patches):
        super(PatchRelativeMultiHeadAttention, self).__init__()
        initializer = keras.initializers.RandomNormal()
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.head_size = head_size
        d_model = num_heads*head_size
        self.wq = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.wk = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.wv = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.dense = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_size).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, head_size)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        patch_embeddings,
        patches_relative_feature,
        patches_relative_mask,
        mask=None,
        layer=0,
        tf_log_dict=None,
    ):
        """
        relative_city_matrix:[batch_size, num_max_cities, vec]
        link_mask:[batch_size, num_max_cities, num_max_cities]
        q_mask:[batch_size, num_max_cities, 1]
        v_mask:[batch_size, num_max_cities, 1]
        """

        log_dict = dict()
        for head in range(self.num_heads):
            log_dict["tf_log_patch_attention_entropy_layer_" + str(layer) + "_head_" + str(head)] = 0

        res = []

        for i in range(self.num_patches):
            # single_unit_q = patch_embeddings[:, i]
            single_unit_q = tf.expand_dims(patch_embeddings[:, i], axis=1)
            # single_unit_q = tf.tile(tf.expand_dims(patch_embeddings[:, i], axis=1), [1, tf.shape(patch_embeddings)[1], 1])
            relative_unit_embeddings = tf.concat([patch_embeddings, patches_relative_feature[:, i]], axis=-1)
            # single_unit_attention_mask = patches_relative_mask[:, i]

            single_unit_q = self.wq(single_unit_q)
            single_unit_k = self.wk(relative_unit_embeddings)
            single_unit_v = self.wv(relative_unit_embeddings)
            # print('single_unit_q', single_unit_q.shape)
            # print('single_unit_k', single_unit_k.shape)

            batch_size = tf.shape(single_unit_q)[0]

            single_unit_q = self.split_heads(single_unit_q, batch_size)
            single_unit_k = self.split_heads(single_unit_k, batch_size)
            single_unit_v = self.split_heads(single_unit_v, batch_size)

            matmul_qk = tf.matmul(single_unit_q, single_unit_k, transpose_b=True)

            # scale matmul_qk
            dk = tf.cast(tf.shape(single_unit_k)[-1], matmul_qk.dtype)
            scaled_attention_scores = matmul_qk / tf.math.sqrt(dk)

            for head in range(self.num_heads):
                log_dict["tf_log_patch_attention_entropy_layer_" + str(layer) + "_head_" + str(head)] += tf.reduce_mean(
                    tf_log_softmax_entropy_with_logits(scaled_attention_scores[:, head, 0])
                )

            # print('matmul_qk', matmul_qk.shape)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
            # add up to 1.
            attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)  # (..., seq_len_q, seq_len_k)

            # (..., seq_len_q, depth_v)
            scaled_attention = tf.matmul(attention_weights, single_unit_v)
            # print('scaled_attention', scaled_attention.shape)

            # (batch_size, seq_len_q, num_heads, head_size)
            scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

            # concat multiple heads
            # (batch_size, seq_len_q, d_model)
            concat_attention = tf.reshape(
                scaled_attention, (-1, tf.shape(scaled_attention)[1], self.num_heads * self.head_size)
            )

            # (batch_size, seq_len_q, d_model)
            single_unit_embeddings = self.dense(concat_attention)

            res.append(single_unit_embeddings)

        output = tf.concat(res, axis=1)

        if tf_log_dict is not None:
            for key in log_dict.keys():
                tf_log_dict[key] = tf.divide(log_dict[key], self.num_patches)

        return output

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        d_model = num_heads * head_size
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.wq = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.wk = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.wv = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        self.dense = keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_size).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, head_size)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, q_mask=None, v_mask=None, unit_length=None, name=None, tf_log_dict=None):
        batch_size = tf.shape(q)[0]
        query_inputs = q

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, head_size)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, head_size)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, head_size)
        v = self.split_heads(v, batch_size)

        scaled_attention_scores = scaled_dot_product_attention_score((q, k))
        if v_mask is not None:
            scaled_attention_scores = apply_mask(
                scaled_attention_scores, tf.reshape(v_mask, (-1, 1, 1, tf.shape(v_mask)[1])), mode="add"
            )

        entropy_masks = tf.squeeze(v_mask, axis=-1)
        for i in range(self.num_heads):
            unit_entropy = 0
            for j in range(unit_length):
                unit_entropy += (
                    tf_log_softmax_entropy_with_logits(scaled_attention_scores[:, i, j]) * entropy_masks[:, j]
                )

            inputs_len = tf.cast(tf.math.count_nonzero(tf.reduce_max(tf.abs(query_inputs), 2), axis=1), tf.float32)
            unit_entropy = tf.math.divide(unit_entropy, inputs_len)

            tf_log_dict["tf_log_" + str(name) + "_entropy_head_" + str(i)] = tf.reduce_mean(unit_entropy)
            tf_log_dict["tf_log_" + str(name) + "_max_weight"] = tf.reduce_max(
                tf.reduce_max(keras.layers.Flatten()(scaled_attention_scores), axis=-1), axis=-1
            )

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_scores, axis=-1)  # (..., seq_len_q, seq_len_k)

        # (..., seq_len_q, depth_v)
        scaled_attention = tf.matmul(attention_weights, v)

        # (batch_size, seq_len_q, num_heads, head_size)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])

        # concat multiple heads
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(
            scaled_attention, (-1, tf.shape(scaled_attention)[1], self.num_heads * self.head_size)
        )

        # (batch_size, seq_len_q, d_model)
        outputs = self.dense(concat_attention)
        if q_mask is not None:
            outputs = apply_mask(outputs, q_mask, mode="mul")
        return outputs


def dot_product_attention_score(inputs):
    q, k = inputs
    score = tf.matmul(q, k, transpose_b=True)
    return score


def scaled_dot_product_attention_score(inputs):
    """Calculate the attention score.
    Args:
        inputs: List/Tuple (q, k)
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
    Returns:
        scaled_attention_scores: (..., seq_len_q, seq_len_k)
    """
    q, k = inputs

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], matmul_qk.dtype)
    scaled_attention_scores = matmul_qk / tf.math.sqrt(dk)
    return scaled_attention_scores


class GeneralAttentionScore(keras.layers.Layer):
    def __init__(self, attention_size):
        super(GeneralAttentionScore, self).__init__()
        self._w = None
        self.attention_size = attention_size

    def build(self, input_shape):
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self._w = keras.layers.Dense(1, kernel_initializer=initializer, activation=None)

    def call(self, inputs, action_name):
        q, k = inputs
        q = tf.tile(tf.expand_dims(q, axis=1), [1, tf.shape(k)[1], 1])
        score = self._w(tf.concat([q, k], axis=-1))
        score = tf.squeeze(score, axis=-1)
        # if action_name == 'select_attack_city_cmd':
        #     print('q', q)
        #     print('score', score)

        return score


class ConcatAttentionScore(keras.layers.Layer):
    def __init__(self, attention_size):
        super(ConcatAttentionScore, self).__init__()
        self._wq, self._wk, self._v = None, None, None
        self.attention_size = attention_size

    def build(self, input_shape):
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self._wq = keras.layers.Dense(
            self.attention_size, kernel_initializer=initializer, use_bias=False, activation=None
        )
        self._wk = keras.layers.Dense(
            self.attention_size, kernel_initializer=initializer, use_bias=False, activation=None
        )
        self._v = self.add_weight(
            name="v",
            shape=(self.attention_size,),
            initializer=keras.initializers.VarianceScaling(scale=2.0, distribution="normal"),
        )

    def call(self, inputs, action_name):
        q, k = inputs
        q = tf.expand_dims(q, 1)
        q = self._wq(q)
        k = self._wk(k)
        out = tf.nn.tanh(q + k)
        score = tf.tensordot(out, self._v, axes=1)
        return score


class DenseConcatAttentionScore(keras.layers.Layer):
    def __init__(self, attention_size):
        super(DenseConcatAttentionScore, self).__init__()
        self._wq, self._wk, self._v = None, None, None
        self.attention_size = attention_size

    def build(self, input_shape):
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self._w = keras.layers.Dense(self.attention_size, kernel_initializer=initializer, activation=tf.nn.tanh)

        self._v = self.add_weight(
            name="v",
            shape=(self.attention_size,),
            initializer=keras.initializers.VarianceScaling(scale=2.0, distribution="normal"),
        )

    def call(self, inputs, action_name):
        q, k = inputs
        q = tf.tile(tf.expand_dims(q, axis=1), [1, tf.shape(k)[1], 1])
        out = self._w(tf.concat([q, k], axis=-1))
        score = tf.tensordot(out, self._v, axes=1)
        return score


class AttentionScoreFactory:
    @staticmethod
    def get(method="dot_product"):
        if method == "dot_product":
            return dot_product_attention_score
        elif method == "general":
            return GeneralAttentionScore()
        elif method == "concat":
            return ConcatAttentionScore()
        elif method == "scale_dot_product":
            return scaled_dot_product_attention_score
        else:
            raise ValueError(f"method {method} is not supported")
