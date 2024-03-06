from tensorflow import keras


class Aggregator(keras.layers.Layer):
    """用于聚合多个encoder的输出, 如果有rnn的话，通常放在这个组件"""

    def call(self, inputs, initial_state=None, seq_len=1):
        """
        Args:
            inputs(list(tf.Tensor/np.ndarray)): 长度不限的list
            initial_state(tf.Tensor/None): rnn中的initial state
            seq_len(int): 序列长度，用rnn时才有意义，通常设为1
        Returns:
            outputs(tf.Tensor): 维度为[batch_size, n]
            state(Option<tf.Tensor>): 可选返回，可以返回一个Tensor，也可以
                返回None。state对应rnn中的state，通常，没有用到rnn的时候，返回None。
        """
        raise NotImplementedError
