import tensorflow as tf
from .attention import MultiHeadAttention
from ..utils import Mask
from tensorflow import keras


class MacaronTransformerBlock(keras.layers.Layer):
    """
    reference: https://arxiv.org/pdf/1906.02762.pdf
    """

    def __init__(self, num_head, head_size):
        super(MacaronTransformerBlock, self).__init__()
        self.num_head = num_head
        self.head_size = head_size
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        d_model = num_head * head_size
        self.ff1_dense = keras.layers.Dense(2 * d_model, activation=tf.nn.relu, kernel_initializer=initializer)
        self.ff2_dense = keras.layers.Dense(d_model, kernel_initializer=initializer)
        self.input_layer_norm = keras.layers.LayerNormalization()
        self.multi_head_attention = MultiHeadAttention(num_head, head_size)
        self.attention_layer_norm = keras.layers.LayerNormalization()
        self.ff3_dense = keras.layers.Dense(2 * d_model, activation=tf.nn.relu, kernel_initializer=initializer)
        self.ff4_dense = keras.layers.Dense(d_model, kernel_initializer=initializer)
        self.output_layer_norm = keras.layers.LayerNormalization()

    def call(self, inputs, inputs_len=None):
        """apply multi-head attention to one type of units,
        together with a residual connection and layer normalization
        注意：为了方便residual连接，需要设置 num_head * head_size 等于 tf.shape(inputs)[-1]
        inputs: [None, None, num_head * head_size]
        outputs: [None, None, num_head * head_size]
        """
        # feed-forward layers
        ff_outputs = self.ff1_dense(inputs)
        ff_output = self.ff2_dense(ff_outputs)
        # residual connection and layer normalization
        inputs = self.input_layer_norm(inputs + ff_output * 0.5)
        if inputs_len is not None:
            inputs = Mask(inputs, inputs_len, mode="mul")

        outputs = self.multi_head_attention(inputs, inputs, inputs, inputs_len, inputs_len)
        # residual connection and layer normalization (只对最后一维执行layer_normalize)
        outputs = self.attention_layer_norm(inputs + outputs)

        ff_outputs = self.ff3_dense(outputs)
        ff_outputs = self.ff4_dense(ff_outputs)
        # residual connection and layer normalization
        outputs = self.output_layer_norm(outputs + ff_outputs * 0.5)
        if inputs_len is not None:
            outputs = Mask(outputs, inputs_len, mode="mul")
        return outputs


class MacaronTransformer(keras.layers.Layer):
    def __init__(self, num_block=2, num_head=2, head_size=64):
        super(MacaronTransformer, self).__init__()
        self._dense = keras.layers.Dense(num_head * head_size)
        self._transformer_blocks = []
        for _ in range(num_block):
            self._transformer_blocks.append(MacaronTransformerBlock(num_head, head_size))

    def call(self, inputs, inputs_len=None):
        outputs = self._dense(inputs)
        for transformer_block in self._transformer_blocks:
            outputs = transformer_block(outputs, inputs_len)
        return outputs
