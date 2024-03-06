import tensorflow as tf
from tensorflow import keras


class ValueApproximator(keras.layers.Layer):
    def __init__(self, hidden_layer_sizes, num_heads):
        super(ValueApproximator, self).__init__()
        self._dense_sequence = keras.Sequential()
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        for layer_size in hidden_layer_sizes:
            self._dense_sequence.add(
                keras.layers.Dense(layer_size, kernel_initializer=initializer, activation=tf.nn.relu)
            )
        self._dense_sequence.add(keras.layers.Dense(num_heads, kernel_initializer=initializer))

    def call(self, inputs):
        value = self._dense_sequence(inputs)
        return value
