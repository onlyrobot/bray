import tensorflow as tf
from .aggregator import Aggregator
from .utils import batch_to_seq
from .utils import reduce_rnn_state
from .utils import seq_to_batch
from tensorflow import keras


class GRUAggregator(Aggregator):
    def __init__(self, hidden_layer_sizes, state_size, output_size):
        super(GRUAggregator, self).__init__()
        self._state_size = state_size
        self._dense_sequence = keras.Sequential()
        for layer_size in hidden_layer_sizes:
            self._dense_sequence.add(keras.layers.Dense(layer_size, activation="relu"))
        self._gru = keras.layers.GRU(state_size, return_state=True, return_sequences=True)
        self._final_dense = keras.layers.Dense(output_size, activation="relu")

    def get_initial_state(self):
        return tf.zeros(self._state_size)

    def call(self, inputs, initial_state, seq_len=1):
        concat_features = tf.concat(inputs, axis=-1)
        features = self._dense_sequence(concat_features)
        features = batch_to_seq(features, seq_len)
        if seq_len > 1:
            initial_state = reduce_rnn_state(initial_state, seq_len)
        outputs, state = self._gru(features, initial_state=initial_state)
        outputs = seq_to_batch(outputs)
        outputs = self._final_dense(outputs)
        return outputs, state
