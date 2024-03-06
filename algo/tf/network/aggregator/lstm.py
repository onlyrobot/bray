import tensorflow as tf
from .aggregator import Aggregator
from .utils import batch_to_seq
from .utils import reduce_rnn_state
from .utils import seq_to_batch
from tensorflow import keras


class LSTMAggregator(Aggregator):
    def __init__(self, hidden_layer_sizes, state_size, output_size):
        super(LSTMAggregator, self).__init__()
        self._state_size = state_size
        self._dense_sequence = keras.Sequential()
        for layer_size in hidden_layer_sizes:
            self._dense_sequence.add(keras.layers.Dense(layer_size, activation="relu"))
        self._lstm = keras.layers.LSTM(state_size, return_sequences=True, return_state=True)
        self._final_dense = keras.layers.Dense(output_size, activation="relu")

    def get_initial_state(self):
        return tf.zeros(self._state_size * 2)

    def call(self, inputs, initial_state, seq_len=1):
        memory_state, carry_state = tf.split(initial_state, num_or_size_splits=2, axis=1)
        concat_features = tf.concat(inputs, axis=-1)
        features = self._dense_sequence(concat_features)
        features = batch_to_seq(features, seq_len)
        if seq_len > 1:
            memory_state = reduce_rnn_state(memory_state, seq_len)
            carry_state = reduce_rnn_state(carry_state, seq_len)
        outputs, memory_state, carry_state = self._lstm(features, initial_state=[memory_state, carry_state])
        outputs = seq_to_batch(outputs)
        outputs = self._final_dense(outputs)
        return outputs, tf.concat([memory_state, carry_state], axis=1)
