import tensorflow as tf

def batch_to_seq(h, seq_len):
    """
    将tensor h的shape由(batch_size, feature_size)转换成(batch_size/seq_len, seq_len, feature_size)
    :param h: inputs of shape (batch_size, feature_size)
    :param seq_len: number of steps in a sequence
    :return:
    """
    feature_size = h.get_shape()[-1] if len(h.get_shape().as_list()) > 1 else 1
    return tf.reshape(h, [-1, seq_len, feature_size])


def seq_to_batch(h):
    """
    将tensor h的shape由(batch_size, seq_len, feature_size)转换成(batch_size * seq_len, feature_size)
    :param h: tensor of shape (batch_size, seq_len, feature_size)
    :return: tensor of shape (batch_size * seq_len, feature_size)
    """
    feature_size = h.get_shape()[-1]
    return tf.reshape(h, [-1, feature_size])


def reduce_rnn_state(h, seq_len):
    feature_size = h.get_shape()[-1] if len(h.get_shape().as_list()) > 1 else 1
    seq_rnn_state = tf.reshape(h, [-1, seq_len, feature_size])
    return seq_rnn_state[:, 0, ...]
