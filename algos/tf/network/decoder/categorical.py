from functools import partial

import tensorflow as tf
from ..distribution import Categorical
from .decoder import Decoder
from tensorflow import keras

class CategoricalDecoder(Decoder):
    def __init__(self, n, hidden_layer_sizes, activation="relu", temperature=1.0):
        super(CategoricalDecoder, self).__init__()
        self._n = n
        self._temperature = temperature
        self._dense_sequense = keras.Sequential()
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        for layer_size in hidden_layer_sizes:
            self._dense_sequense.add(
                keras.layers.Dense(layer_size, kernel_initializer=initializer, activation=activation)
            )
        self._dense_sequense.add(keras.layers.Dense(n, kernel_initializer=initializer))
        self._embedding_vocabulary = None

    @property
    def distribution_fn(self):
        return partial(Categorical, temperature=self._temperature)

    def build(self, input_shape):
        # self._embedding_vocabulary = keras.layers.Embedding(self._n, input_shape[0][-1])
        super(CategoricalDecoder, self).build(input_shape)

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

        logits = self._dense_sequense(auto_regressive_embedding)

        if action_mask is not None:
            logits = logits - (1 - action_mask) * 1e12
        distribution = Categorical(logits, temperature=self._temperature)

        if behavior_action is None:
            behavior_action = distribution.sample()
        behavior_action = tf.cast(behavior_action, tf.int32)
        # behavior_action_embedding = self._embedding_vocabulary(behavior_action)
        # auto_regressive_embedding = behavior_action_embedding + inputs
        return logits, behavior_action, auto_regressive_embedding
