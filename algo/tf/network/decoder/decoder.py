from tensorflow import keras


class Decoder(keras.layers.Layer):
    """通常作为整个网络的出口，可以存在多个decoder。Decoder的主要作用是
    作出决策，输出action。
    当存在多个decoder的时候，decoder决策时可以condition on其他decoder的决策。
    比如有这样一个场景，将两个小球分别移动到两个指定的地点，一个decoder负责选取小球，
    另一个decoder负责选取目标地点，假如先选小球，那选取目标地点时就应该考虑选了那个小球。
    call方法中返回auto_regressive_embeddings就是为了实现这个功能
    """

    @property
    def distribution_fn(self):
        """
        Returns:
            distrubution_fn: fn(logits) -> Distribution
        """
        raise NotImplementedError

    def call(self, inputs, action_mask=None, behavior_action=None, auto_regressive=False):
        """
        Args:
            inputs(list(tf.Tensor/np.ndarray)): 长度为2的list
            action_mask(tf.Tensor/np.ndarray): mask掉不能执行的action
            behavior_action(tf.Tensor/np.ndarray): 当输入为inputs时，之前的
                策略作出的决策
        Returns:
            logits(tf.Tensor): 用于之后计算loss
            action(tf.Tensor): 作出的决策
            auto_regressive_embeddings(tf.Tensor): 用于多个decoder之间信息传递
        """
        raise NotImplementedError
