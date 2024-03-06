from .encoder import Encoder
from ..encoder import encoder_pooling
from ..utils import length
from ..utils import Mask
from tensorflow import keras


class EntityEncoder(Encoder):
    def __init__(self, input_name, output_name, hidden_layer_sizes, transformer=None, pooling=None):
        super(EntityEncoder, self).__init__()
        self.input_name = input_name
        self.output_name = output_name
        if pooling is not None:
            self._pooling = pooling
        else:
            self._pooling = encoder_pooling.Max()
        self._transformer = transformer
        self._dense_sequence = keras.Sequential()
        initializer = keras.initializers.GlorotNormal()
        for layer_size in hidden_layer_sizes:
            self._dense_sequence.add(keras.layers.Dense(layer_size, kernel_initializer=initializer, activation="relu"))

    def call(self, inputs_dict):
        inputs = inputs_dict[self.input_name]
        inputs_len = length(inputs)
        entity_embeddings = self._dense_sequence(inputs)
        entity_embeddings = Mask(entity_embeddings, inputs_len)
        if self._transformer is not None:
            entity_embeddings = self._transformer(entity_embeddings, inputs_len)
        outputs = self._pooling(entity_embeddings, inputs_len)
        return outputs, entity_embeddings
