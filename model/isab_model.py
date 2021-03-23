from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from layer.transformer.isab import InducedSetAttentionBlock as ISAB
from layer.encodec import get_encoder
from layer.simclr_loss import simclr_loss
from util.eval import hook


class ISABModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.isab = ISAB(conf.d_model, 1, dff=conf.d_model)

        self.encoder = get_encoder(conf)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation=tf.nn.relu),
            tf.keras.layers.Dense(4096)
        ])
        self.context = tf.Variable(tf.initializers.GlorotUniform()([conf.k, conf.d_model]), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)
        feat_1, feat_2 = tf.split(feat, 2, axis=0)
        x_1 = feat_1[tf.newaxis, :, :]  # [1 N D]
        x_2 = feat_2[tf.newaxis, :, :]  # [1 N D]
        i = self.context[tf.newaxis, :, :]  # [1 K D]

        h_1, out_1, att_kn_1, att_nk_1 = self.isab(x_1, i)
        h_2, out_2, att_kn_2, att_nk_2 = self.isab(x_2, i)

        isab_out = tf.concat([out_1, out_2], axis=0)
        a_nk = tf.concat([att_nk_1, att_nk_2], axis=0)
        assign_nk = tf.argmax(a_nk, axis=1)

        a_kn = tf.concat([att_kn_1, att_kn_2], axis=1)
        assign_kn = tf.argmax(a_kn, axis=0)

        pred = self.decoder(isab_out, training=training)

        if training:
            loss_ae = tf.reduce_mean(tf.square(pred - inputs))

            loss_clr, _, _ = simclr_loss(h_1, h_2, self.conf.temp)

            loss = loss_ae + loss_clr

            self.add_loss(loss)
            if step > 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_ae', loss_ae, step=step)
                tf.summary.scalar('loss_clr', loss_clr, step=step)

                tf.summary.histogram('assign_kn', assign_kn, step=step)
                tf.summary.histogram('assign_nk', assign_nk, step=step)

                tf.summary.image('img_kn', a_kn[tf.newaxis, :, :, tf.newaxis], step=step)
                tf.summary.image('img_nk', a_nk[tf.newaxis, :, :, tf.newaxis], step=step)
        return assign_kn, assign_nk, feat


def step_train(conf, data_1: dict, data_2: dict, model: tf.keras.Model, opt: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    feat_2 = data_2['image']

    label_1 = data_1['label']
    label_2 = data_2['label']

    feat_1 = tf.concat([feat_1, feat_2], axis=0)
    label_1 = tf.concat([label_1, label_2], axis=0)

    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        assign_kn, assign_nk, _feat_1 = model(feat_1, step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

    if _step > 0:
        label = label_1
        pred = assign_nk
        feat = feat_1

        acc, nmi, ari, sc = hook(feat.numpy(), label.numpy(), pred.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)
        tf.summary.histogram('eval/assign', pred, step)

        pred = assign_kn
        acc, nmi, ari, sc = hook(feat.numpy(), label.numpy(), pred.numpy())

        tf.summary.scalar('eval2/nmi', nmi, step)
        tf.summary.scalar('eval2/acc', acc, step)
        tf.summary.scalar('eval2/ari', ari, step)
        tf.summary.scalar('eval2/sc', sc, step)
        tf.summary.histogram('eval2/assign', pred, step)

    return loss.numpy()
