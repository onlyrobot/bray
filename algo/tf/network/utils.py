import tensorflow as tf


def get_message(input, tf_log_dict, name, action_mask=None):
    if tf_log_dict is not None:
        if action_mask is None:
            tran_input = tf.transpose(input, perm=[1, 0])
            mean = tf.reduce_mean(tran_input, axis=-1, keepdims=True)
            message = tf.reduce_mean(tf.abs(tran_input - mean), axis=-1)
        else:
            tran_input = tf.transpose(input, perm=[1, 0])
            tran_action_mask = tf.transpose(action_mask, perm=[1, 0])
            sum = tf.reduce_sum(tran_input*tran_action_mask, axis=-1, keepdims=True)
            count = tf.reduce_sum(tran_action_mask, axis=-1, keepdims=True) + 1e-5
            mean = sum/count
            message = tf.reduce_mean(tf.abs(tran_input - mean)*tran_action_mask, axis=-1)
            tf_log_dict["tf_log_" + name + "_max_axis_message"] = tf.reduce_max(message)
            tf_log_dict["tf_log_" + name + "_min_axis_message"] = tf.reduce_min(message)
        tf_log_dict["tf_log_" + name + "_message"] = tf.reduce_mean(message)

def get_mask(inputs):
    """支持任意形状mask，非零单位不一定需要在最前面"""
    mask = tf.cast(tf.greater(tf.reduce_max(tf.abs(inputs), axis=-1, keepdims=True), 0.0), tf.float32)
    return mask


def apply_mask(inputs, mask=None, mode="mul"):
    if mask is None:
        return inputs
    else:
        if mode == "mul":
            return inputs * mask
        if mode == "add":
            return inputs - (1 - mask) * 1e12


def length(inputs):
    # 确认每个sample中包含的实际active的unit数量，即非全0行
    inputs_len = tf.math.count_nonzero(tf.reduce_max(tf.abs(inputs), 2), axis=1)
    # 如果全部全0行，保留1行
    # inputs_len = tf.maximum(inputs_len, 1)
    return inputs_len


def Mask(inputs, inputs_len, mode="mul", seq_axis=1):
    """
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, inputs_len, input_size)的张量；
    inputs_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    """
    assert inputs_len is not None, "inputs_len should not be None"
    inputs_shape = tf.shape(inputs)
    max_len = inputs_shape[seq_axis]
    mask = tf.sequence_mask(inputs_len, max_len, inputs.dtype)
    mask_shape = tf.shape(mask)
    mask_new_shape = [1] * inputs_shape.get_shape()[0]
    mask_new_shape[0] = mask_shape[0]
    mask_new_shape[seq_axis] = mask_shape[1]
    mask = tf.reshape(mask, mask_new_shape)
    if mode == "mul":
        outputs = inputs * mask
    elif mode == "add":
        outputs = inputs - (1 - mask) * 1e12
    else:
        raise ValueError(f"mode {mode} is not supported")
    return outputs