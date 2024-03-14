from .encoder import Encoder
from tensorflow import keras


class CommonEncoder(Encoder):
    def __init__(self, input_name, output_name, hidden_layer_sizes):
        super(CommonEncoder, self).__init__()
        self.input_name = input_name
        self.output_name = output_name
        self._dense_sequence = keras.Sequential()
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        # keras.initializers.GlorotNormal()
        for layer_size in hidden_layer_sizes:
            self._dense_sequence.add(keras.layers.Dense(layer_size, kernel_initializer=initializer, activation="relu"))

    def call(self, inputs_dict, tf_log_dict):
        input = inputs_dict[self.input_name]
        return self._dense_sequence(input), None
