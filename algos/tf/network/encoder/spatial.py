import tensorflow as tf
from .encoder import Encoder
from ..layer.resnet import Block, GN
from tensorflow import keras
from tensorflow.keras.layers import MaxPooling2D

class SpatialEncoder(Encoder):
    """用于处理空间特征信息
    输入为[batch_size, height, width, channels]

    Attributes:
        projected_channel_num: Integer, Dense层将输入变成[batch_size, height, width, projected_channel_num]
        output_size: Integer, Dense层将输出变成[batch_size, output_size]
        down_samples: List[Tuple(filters, kernel_size, strides)], 对空间特征进行下采样
        res_block_num: Integer, 添加残差网络层对空间特征进行提取
    """

    def __init__(self, input_name, output_name, projected_channel_num, output_size, down_samples=None, resnet_config=None):
        super(SpatialEncoder, self).__init__()

        self.input_name = input_name
        self.output_name = output_name
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self._layers = keras.Sequential()
        if projected_channel_num is not None:
            self._layers.add(keras.layers.Dense(
                projected_channel_num, kernel_initializer=initializer, activation="relu"
            ))
            self._layers.add(GN(projected_channel_num, projected_channel_num))
            self._layers.add(tf.keras.layers.ReLU())

        if down_samples:
            for (filters, kernel_size, strides, padding, pooling) in down_samples:
                self._layers.add(
                    keras.layers.Conv2D(
                        filters,
                        kernel_size,
                        strides=strides,
                        padding=padding,
                        kernel_initializer=initializer,
                        activation="relu",
                        data_format="channels_last",
                    )
                )
                self._layers.add(GN(filters, filters))
                self._layers.add(tf.keras.layers.ReLU())
                if pooling == "pooling":
                    self._layers.add(MaxPooling2D(pool_size=(2, 2)))
        if resnet_config is not None:
            for i, channel in enumerate(resnet_config['channels']):
                if i > 0:
                    self._layers.add(Block(channel, True)) 
                for _ in range(3):
                    self._layers.add(Block(channel, False))

        self._outputs_dense = keras.layers.Dense(output_size, kernel_initializer=initializer, activation="relu")

    def call(self, inputs_dict, tf_log_dict):
        """
        Args:
            inputs: np.ndarray/tf.Tensor, [batch_size, height, width, channels]
        Returns:
            outputs: tf.Tensor, [batch_size, output_size]
            spatial_embeddings: tf.Tensor,
                if down_samples: [batch_size, down_sampled_height, down_sampled_width, last_down_sample_filters]
                else: [batch_size, height, width, projected_channel_num]
        """
        inputs = inputs_dict[self.input_name]

        spatial_embeddings = self._layers(inputs)
        tf_log_dict[f'max_embedding_in_{self.output_name}_encoder'] = tf.reduce_max(spatial_embeddings)

        if len(spatial_embeddings.shape) == 5:
            spatial_embeddings = tf.reshape(
                spatial_embeddings, [spatial_embeddings.shape[0], spatial_embeddings.shape[1], -1]
            )
            outputs = self._outputs_dense(spatial_embeddings)
            return outputs, outputs

        elif len(spatial_embeddings.shape) == 4:
            outputs = self._outputs_dense(keras.layers.Flatten()(spatial_embeddings))
            return outputs, spatial_embeddings
