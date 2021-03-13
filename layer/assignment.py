from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.transformer.isab import MultiheadAttentionBlock


class Assigner(tf.keras.layers.Layer):
    def __init__(self, conf):
        """
        A dummy class for assigners
        We extend a basic keras layer with two requirements:
        - One needs to define an additional loss term in case it is needed
        - Each single call returns two tensors for the aggregated features and the cluster assignment
        :param conf:
        """
        super().__init__()
        self.conf = conf

    def _additional_loss(self, *args, **kwargs):
        return 0

    # noinspection PyMethodOverriding
    def call(self, context, sample, training=True):
        n = tf.shape(sample)[0]
        k = tf.shape(context)[0]
        return context, tf.zeros([n, k])


class SoftAssigner(Assigner):
    def __init__(self, conf):
        super().__init__(conf)
        self.mab = MultiheadAttentionBlock(self.conf.d_model, 1, self.conf.d_model * 2)

    def call(self, context, sample, training=True):
        """

        :param context: [K D]
        :param sample: [N D]
        :param training:
        :return:
        """
        _context = tf.expand_dims(context, axis=0)
        _sample = tf.expand_dims(sample, axis=0)
        agg_feat, assignment = self.mab(_context, _sample, training=training)
        agg_feat = tf.squeeze(agg_feat)
        assignment = tf.squeeze(assignment)

        return agg_feat, assignment


def get_assigner(conf) -> Assigner:
    cases = {
        'soft': SoftAssigner
    }
    Rslt = cases.get(conf.assigner)

    return Rslt(conf)
