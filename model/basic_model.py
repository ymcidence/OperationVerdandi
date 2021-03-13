from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.encodec import get_encoder
from layer.assignment import get_assigner


class BasicModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.encoder = get_encoder(conf)
        self.assigner = get_assigner(conf)

        self.context = tf.Variable(initial_value=tf.random.normal([conf.k, conf.d_model], stddev=.01), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')

    def call(self, inputs, training=True, mask=None):
        feat = self.encoder(inputs, training=training)
        agg_feat, assignment = self.assigner(self.context, feat, training=training)

        return agg_feat, assignment


def basic_step(data_1: dict, data_2: dict, model: BasicModel, opt: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    feat_2 = data_2['image']

    agg_1, assign_1 = model(feat_1)
    agg_2, assign_2 = model(feat_2)

    agg_2 = tf.stop_gradient(agg_2)
