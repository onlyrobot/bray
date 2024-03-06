import tensorflow as tf
from ..layer.pooling import AttentionPooling
from tensorflow import keras


class Single:
    def __call__(self, entities_embeddings, selected_entities_one_hot):
        """对指定的单位做mean pooling和max pooling，并做拼接
        Args:
            entities_embeddings: shape == (batch_size, max_num, depth)
            selected_entities_one_hot: shape == (batch_size, max_num)，注意这是一个one hot
        Returns:
            selected_mean_max_embeddings: shape == (batch_size, depth * 2)
        """
        selected_embeddings = entities_embeddings * tf.expand_dims(selected_entities_one_hot, axis=-1)
        res = tf.reduce_sum(selected_embeddings, axis=1)
        return res


class MeanMax:
    def __call__(self, entities_embeddings, selected_entities_one_hot):
        """对指定的单位做mean pooling和max pooling，并做拼接
        Args:
            entities_embeddings: shape == (batch_size, max_num, depth)
            selected_entities_one_hot: shape == (batch_size, max_num)，注意这是一个one hot
        Returns:
            selected_mean_max_embeddings: shape == (batch_size, depth * 2)
        """
        selected_embeddings = entities_embeddings * tf.expand_dims(selected_entities_one_hot, axis=-1)
        selected_count = tf.maximum(1.0, tf.reduce_sum(selected_entities_one_hot, axis=-1, keepdims=True))
        selected_mean_embeddings = tf.reduce_sum(selected_embeddings, axis=1) / selected_count
        selected_max_embeddings = tf.reduce_max(selected_embeddings, axis=1)
        selected_mean_max_embeddings = tf.concat([selected_mean_embeddings, selected_max_embeddings], axis=-1)
        return selected_mean_max_embeddings


class Attention(keras.layers.Layer):
    def __init__(self, num_query=4, num_head=4, head_size=8, min_mask_val=0.1):
        super(Attention, self).__init__()
        self._min_mask_val = min_mask_val
        self._attention_pooling = AttentionPooling(num_query, num_head, head_size)

    def call(self, entities_embeddings, selected_entities):
        selected_entites_mask = tf.expand_dims(tf.maximum(self._min_mask_val, selected_entities), axis=-1)
        selected_entities_embeddings = selected_entites_mask * entities_embeddings
        attention_embeddings = self._attention_pooling(selected_entities_embeddings)
        return attention_embeddings
