import tensorflow as tf
from .encoder import Encoder


class CombinedEntityEncoder(Encoder):
    def __init__(self, encoder_list, output_name):
        super(CombinedEntityEncoder, self).__init__()
        self._encoder_list = encoder_list
        self.output_name = output_name

    def call(self, inputs_dict):
        encoder_output = []
        for encoder in self._encoder_list:
            _, embeddings = encoder(inputs_dict)
            encoder_output.append(embeddings)
        return None, tf.concat(encoder_output, axis=2)
