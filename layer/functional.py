from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


@tf.function
def row_distance(tensor_a, tensor_b):
    """
    :param tensor_a: [N1 D]
    :param tensor_b: [N2 D]
    :return: [N1 N2]
    """
    na = tf.reduce_sum(tf.square(tensor_a), 1)
    nb = tf.reduce_sum(tf.square(tensor_b), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    rslt = tf.sqrt(tf.maximum(na - 2 * tf.matmul(tensor_a, tensor_b, False, True) + nb, 0.0))

    return rslt


@tf.function
def row_distance_cosine(tensor_a, tensor_b):
    norm_a = tf.sqrt(tf.reduce_sum(tf.pow(tensor_a, 2), 1, keepdims=True))  # [N, 1]
    norm_b = tf.sqrt(tf.reduce_sum(tf.pow(tensor_b, 2), 1, keepdims=True))
    denominator = tf.matmul(norm_a, norm_b, transpose_b=True)
    numerator = tf.matmul(tensor_a, tensor_b, transpose_b=True)

    return numerator / denominator


def nearest_context(feature, context):
    distances = row_distance(feature, context)
    min_ind = tf.cast(tf.argmin(distances, axis=1), dtype=tf.int32)
    k = tf.shape(context)[0]
    min_ind = tf.one_hot(min_ind, k, dtype=tf.float32)  # [N k]
    rslt = min_ind @ context
    return rslt, tf.stop_gradient(min_ind)


@tf.custom_gradient
def vq(feature, context):
    value, ind = nearest_context(feature, context)

    def grad(d_value):
        d_context = tf.matmul(ind, d_value, transpose_a=True)
        return d_value, d_context

    return value, grad


@tf.function
def build_adjacency(feature):
    """

    :param feature: [N d]
    :return:
    """
    adj = tf.nn.relu(row_distance_cosine(feature, feature))
    adj = tf.pow(adj, 1)
    return adj


@tf.function
def build_adjacency_v1(feature, t=.1):
    """

    :param feature: [N d]
    :param t: temperature
    :return:
    """

    squared_sum = tf.reshape(tf.reduce_sum(feature * feature, 1), [-1, 1])
    distances = squared_sum - 2 * tf.matmul(feature, feature, transpose_b=True) + tf.transpose(squared_sum)
    adjacency = tf.exp(-1 * distances / t)
    return adjacency


class Dummy(tf.keras.layers.Layer):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def call(self, inputs, **kwargs):
        return self.f(inputs)
