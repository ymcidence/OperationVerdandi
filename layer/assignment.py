from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.transformer.isab import MultiheadAttentionBlock
from layer.gumbel import gumbel_softmax
from layer.gcn import GCNLayer


class Assigner(tf.keras.Model):
    # noinspection PyMethodOverriding
    def call(self, context, sample, training=True):
        raise NotImplementedError


class AttentionAssigner(Assigner):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.mab = MultiheadAttentionBlock(self.conf.d_model, 1, self.conf.d_model * 2)

    # noinspection PyMethodOverriding
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


class SoftAssigner(Assigner):
    def __init__(self, conf):
        super().__init__()

        self.v = tf.keras.layers.Dense(conf.d_model)  # [N D]
        self.k = tf.keras.layers.Dense(conf.d_model)  # [N D]
        self.q = tf.keras.layers.Dense(conf.d_model)  # [K D]

        self.ln = tf.keras.layers.LayerNormalization()
        self.gumbel_temp = conf.gumbel_temp
        self.d_model = conf.d_model

    def call(self, context, sample, training=True, step=-1):
        _q = self.q(context)  # [K D]
        _k = self.k(sample)  # [N D]
        _v = self.v(sample)  # [N D]

        _qk = tf.matmul(_q, _k, transpose_b=True)  # [K N]

        if self.gumbel_temp > 0:  # when the gumbel trick is used
            assignment = gumbel_softmax(tf.transpose(_qk), self.gumbel_temp, hard=False)  # [N K]
            _assignment = gumbel_softmax(tf.transpose(_qk), self.gumbel_temp, hard=True)
            agg_feat = tf.matmul(assignment, _v, transpose_a=True)  # [K D]
            # sum_energy = 1 / (tf.reduce_sum(assignment, axis=0) + 1e-8)

            # agg_feat = tf.einsum('kd,k->kd', agg_feat, sum_energy)

            agg_feat = self.ln(agg_feat, training=training)

            if step > 0:
                tf.summary.image('att', _qk[tf.newaxis, :, :, tf.newaxis], step=step)

            return agg_feat, _assignment

        else:
            raise NotImplementedError


class GCNAssigner(Assigner):
    def __init__(self, conf):
        super().__init__()
        self.gcn = GCNLayer(conf.d_model)
        self.projection = tf.keras.layers.Dense(conf.d_model)
        self.k = conf.k
        self.temp = conf.gumbel_temp

    def call(self, context, sample, training=True, step=-1):
        all_samples = tf.concat([context, sample], axis=0)
        projected = self.projection(all_samples)


def get_assigner(conf) -> tf.keras.Model:
    cases = {
        'soft': SoftAssigner
    }
    Rslt = cases.get(conf.assigner)

    return Rslt(conf)
