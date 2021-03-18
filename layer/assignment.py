from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.transformer.isab import MultiheadAttentionBlock
from layer.gumbel import gumbel_softmax
from layer.gcn import GCNLayer
import numpy as np


class Assigner(tf.keras.Model):
    # noinspection PyMethodOverriding
    def call(self, context, sample, training=True):
        raise NotImplementedError

    def plot_helper(self):
        _s = tf.keras.Input(shape=(512), dtype=tf.float32, name='input1')
        _c = tf.keras.Input(shape=(512), dtype=tf.float32, name='input2')
        agg, ass = self.call(_c, _s)
        return tf.keras.Model(inputs=[_c, _s], outputs=[agg, ass])


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
            sum_energy = 1 / (tf.reduce_sum(assignment, axis=0) + 1e-8)

            agg_feat = tf.einsum('kd,k->kd', agg_feat, sum_energy)

            # agg_feat = self.ln(agg_feat, training=training)

            if step > 0:
                tf.summary.image('att', _qk[tf.newaxis, :, :, tf.newaxis], step=step)

            return agg_feat, _assignment

        else:
            raise NotImplementedError


class GCNAssigner(Assigner):
    def __init__(self, conf):
        super().__init__()
        self.gcn = GCNLayer(conf.d_model)
        self.fc_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(conf.d_model),
            tf.keras.layers.BatchNormalization()
        ])
        self.fc_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(conf.d_model),
            tf.keras.layers.BatchNormalization()
        ])
        # self.fc_3 = tf.keras.Sequential([
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Dense(conf.d_model),
        #     tf.keras.layers.BatchNormalization()
        # ])
        self.k = conf.k
        self.temp = conf.gumbel_temp
        self.d_model = conf.d_model
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, context, sample, training=True, step=-1):
        sample = self.bn(sample, training=training)
        _s = self.fc_1(sample, training=training)
        _c = self.fc_2(context, training=training)
        projected = tf.concat([_c, _s], axis=0)
        all_samples = tf.concat([context, sample], axis=0)
        # all_samples = self.fc_3(all_samples, training=training)

        dist = self.row_distance(projected, projected) / self.d_model  # [N+K N+K]
        all_adj = tf.exp(-dist / self.temp, name='adj') + 1e-8
        gcn_rslt = self.gcn(all_samples, all_adj)

        _k = tf.shape(context)[0]
        _d = tf.shape(context)[1]
        _n = tf.shape(sample)[0]

        # rslt = tf.slice(gcn_rslt, [0, 0], [_k, _d])

        rslt = gcn_rslt  # [:_k, :]
        assignment = tf.slice(all_adj, [_k, 0], [_n, _k])  # [N K]

        if step > 0:
            fig = tf.slice(all_adj, [0, _k], [_k, _n])[tf.newaxis, :, :, tf.newaxis]  # [1 K N 1]
            tf.summary.image('att', fig, step=step)

        # self.add_loss(tf.reduce_mean(gcn_rslt) * 0)

        return rslt, assignment

    @staticmethod
    def row_distance(tensor_a, tensor_b):
        """
        :param tensor_a: [N1 D]
        :param tensor_b: [N2 D]
        :return: [N1 N2]
        """
        na = tf.reduce_sum(tf.square(tensor_a), 1)
        nb = tf.reduce_sum(tf.square(tensor_b), 1)

        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        rslt = na - 2 * tf.matmul(tensor_a, tensor_b, False, True) + nb

        return rslt


def get_assigner(conf) -> tf.keras.Model:
    cases = {
        'soft': SoftAssigner,
        'gcn': GCNAssigner
    }
    Rslt = cases.get(conf.assigner)

    return Rslt(conf)


def test_plot():
    from util.config import parser

    conf = parser.parse_args()
    model = GCNAssigner(conf)

    m = model.plot_helper()
    tf.keras.utils.plot_model(m, to_file='gcn.jpg', show_shapes=True, expand_nested=True, show_layer_names=True)


def test_grad():
    from util.config import parser
    gcn = GCNLayer(512)
    fc = tf.keras.layers.Dense(512)
    _ = parser.parse_args()
    with tf.GradientTape() as tape:
        a = tf.Variable(tf.random.normal([1024, 1024]), trainable=True)
        b = fc(a)
        o = GCNAssigner.row_distance(b, b)
        adj = tf.exp(-o / 32)
        _gcn = gcn(a, adj)

        loss = tf.reduce_mean(_gcn)

        gradients = tape.gradient(target=loss, sources=[a, b])
        print(gradients)

    # inp = tf.keras.Input(shape=(128), dtype=tf.float32, name='input1')
    # ib = fc(inp)
    # io = row_distance(ib, ib)
    # adj = tf.exp(-io / 128)
    # _gcn = gcn(inp, adj)
    # model = tf.keras.Model(inputs=[inp], outputs=[_gcn])
    # tf.keras.utils.plot_model(model, to_file='test.jpg', show_shapes=True, expand_nested=True, show_layer_names=True)


if __name__ == '__main__':
    test_grad()
