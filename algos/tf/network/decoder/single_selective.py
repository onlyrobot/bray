from functools import partial

import tensorflow as tf
from ..distribution import Categorical
from .decoder import Decoder
from .decoder_pooling import Single
from ..layer.attention import ConcatAttentionScore
from ..layer.attention import DenseConcatAttentionScore
from ..layer.attention import GeneralAttentionScore
from tensorflow import keras


class SingleSelectiveDecoder(Decoder):
    def __init__(
        self,
        attention_size=64,
        temperature=1.0,
        decoder_attention_type="general",
        auto_regressive_method="concat",
    ):
        super(SingleSelectiveDecoder, self).__init__()
        self._temperature = temperature
        if decoder_attention_type == "general":
            self._attention_score = GeneralAttentionScore(attention_size)
        elif decoder_attention_type == "concat":
            self._attention_score = ConcatAttentionScore(attention_size)
        elif decoder_attention_type == "dense_concat":
            self._attention_score = DenseConcatAttentionScore(attention_size)
        self._auto_regressive_dense = None
        self._auto_regressive_method = auto_regressive_method
        self.empty_action_embedding = None

    @property
    def distribution_fn(self):
        return partial(Categorical, temperature=self._temperature)

    def build(self, input_shape):
        output_size = input_shape[0][-1]
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        if self._auto_regressive_method == "concat":
            self._auto_regressive_dense = keras.layers.Dense(
                output_size, kernel_initializer=initializer, activation=tf.nn.relu
            )
        elif self._auto_regressive_method == "plus":
            self._auto_regressive_dense = keras.layers.Dense(
                output_size, kernel_initializer=initializer, activation=None
            )
        super(SingleSelectiveDecoder, self).build(input_shape)

    def call(
        self,
        inputs,
        action_mask=None,
        behavior_action=None,
        auto_regressive=False,
        action_name=None,
        decoder_info_dict=None,
        tf_log_dict=None,
    ):
        auto_regressive_embedding, source_embeddings = inputs

        query = auto_regressive_embedding
        key = source_embeddings
        attention_logits = self._attention_score((query, key), action_name)


        logits = attention_logits

        if action_mask is not None:
            logits = logits - (1.0 - action_mask) * 1e12
            # tf.print(action_name, self.empty_action_embedding, summarize=1000)
        distribution = Categorical(logits, self._temperature)

        if behavior_action is None:
            behavior_action = distribution.sample()
        behavior_action = tf.cast(behavior_action, tf.int32)
        behavior_action_one_hot = tf.one_hot(behavior_action, depth=tf.shape(source_embeddings)[1])
        pooling = Single()
        selected_embedding = pooling(source_embeddings, behavior_action_one_hot)

        if auto_regressive:
            if self._auto_regressive_method == "concat":
                auto_regressive_embedding = self._auto_regressive_dense(
                    tf.concat([auto_regressive_embedding, selected_embedding], axis=-1)
                )
            elif self._auto_regressive_method == "plus":
                selected_embedding = self._auto_regressive_dense(selected_embedding)
                auto_regressive_embedding = auto_regressive_embedding + selected_embedding
        else:
            auto_regressive_embedding = auto_regressive_embedding

        if tf_log_dict is not None:
            tf_log_dict["tf_log_attention_logits_message_" + action_name] = tf.reduce_sum(
                tf.abs(attention_logits * action_mask)
            )

        return logits, behavior_action, auto_regressive_embedding
