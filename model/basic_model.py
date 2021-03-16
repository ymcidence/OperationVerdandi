from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.encodec import get_encoder
from layer.assignment import get_assigner
from layer.simclr_loss import simclr_loss
from util.eval import hook


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

        return agg_feat, assignment, feat


def step_train(conf, data_1: dict, data_2: dict, model: BasicModel, opt: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    feat_2 = data_2['image']

    label_1 = data_1['label']
    label_2 = data_2['label']

    _step = -1 if step % 50 > 0 else step

    with tf.GradientTape() as tape:
        agg_1, assign_1, feat_1 = model(feat_1)
        agg_2, assign_2, feat_2 = model(feat_2)
        loss, _, _ = simclr_loss(agg_1, agg_2, conf.temp)

        gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

    if _step > 0:
        label = tf.concat([label_1, label_2], axis=0)
        pred = tf.concat([assign_1, assign_2], axis=0)
        pred = tf.argmax(pred, axis=1)
        feat = tf.concat([feat_1, feat_2], axis=0)

        acc, nmi, ari, sc = hook(feat.numpy(), label.numpy(), pred.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)

        tf.summary.histogram('eval/assign', pred, step)

    return loss.numpy()
