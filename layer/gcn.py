from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

OVERFLOW_MARGIN = 1e-8
import numpy as np


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, fc=True, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.fc = fc
        if fc:
            self.fc = tf.keras.layers.Dense(out_dim)

    # noinspection PyMethodOverriding
    def call(self, values, adjacency, **kwargs):
        """
        :param values:
        :param adjacency:
        :param kwargs:
        :return:
        """
        return self.spectrum_conv(values, adjacency)

    def spectrum_conv(self, values, adjacency):
        """
        Convolution on a graph with graph Laplacian
        :param values: [N D]
        :param adjacency: [N N] must be self-connected
        :return:
        """
        fc_sc = self.fc(values) if self.fc else values
        conv_sc = self.graph_laplacian(adjacency) @ fc_sc
        return conv_sc

    @staticmethod
    def graph_laplacian(adjacency):
        """
        :param adjacency: must be self-connected
        :return:
        """
        graph_size = tf.shape(adjacency)[0]
        d = adjacency @ tf.ones([graph_size, 1])
        d_inv_sqrt = 1 / (tf.pow(d, 0.5) + OVERFLOW_MARGIN)
        d_inv_sqrt = tf.eye(graph_size) * d_inv_sqrt
        laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt
        return laplacian
