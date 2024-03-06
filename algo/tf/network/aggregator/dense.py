import tensorflow as tf
from .aggregator import Aggregator
from tensorflow import keras


class DenseAggregator(Aggregator):
    def __init__(self, output_size):
        super(DenseAggregator, self).__init__()
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self._dense = keras.layers.Dense(output_size, kernel_initializer=initializer, activation="relu")

    def call(
        self,
        inputs,
        initial_state=None,
        seq_len=1,
    ):
        del initial_state, seq_len  # Unused by call
        concat_features = tf.concat(inputs, axis=-1)

        outputs = self._dense(concat_features)
        return outputs, None
