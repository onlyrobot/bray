from tensorflow import keras


class Encoder(keras.layers.Layer):
    """Encoder通常作为整个网络的入口，接收state数据，将其处理成[batch_size, n]的向量"""

    def call(self, inputs):
        """
        Args:
            inputs(tf.Tensor/np.ndarray): 维度[batch_size, ...]
        Returns:
            outputs(tf.Tensor): 维度为[batch_size, n]
            embeddings(Option<tf.Tensor>): 这是一个可选返回，可以返回一个Tensor，
                也可以返回None。比如说当输入是一张图像的时候，维度为[batch_size, height, width, channel],
                不仅想返回outputs，还想输出一个保留空间信息的embedding。
        """
        raise NotImplementedError
