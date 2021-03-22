from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from layer.encodec import get_encoder
from layer.gcn import GCNLayer
from layer.functional import vq, row_distance
from layer.gumbel import gumbel_softmax
from util.eval import hook


class AEModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.encoder = get_encoder(conf)

        self.context = tf.Variable(initial_value=tf.random.normal([conf.k, conf.d_model]), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')
        self.k = conf.k
        self.conf = conf

        # self.gcn = GCNLayer(conf.d_model)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2048),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(4096),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)
        vq_feat = vq(feat, self.context)

        pred = self.decoder(vq_feat, training=training)

        assignment = -tf.stop_gradient(row_distance(feat, self.context))

        ind = tf.argmax(assignment, axis=1)
        one_hot = tf.one_hot(ind, self.k)

        if training:
            indexed_emb = one_hot @ self.context
            loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(pred - inputs), axis=1) / 2.)
            loss_vq_1 = tf.reduce_mean(tf.reduce_mean(tf.square(tf.stop_gradient(feat) - indexed_emb), axis=1) / 2.)
            loss_vq_2 = tf.reduce_mean(tf.reduce_mean(tf.square(tf.stop_gradient(indexed_emb) - feat), axis=1) / 2.)

            loss = loss_ae + loss_vq_1 + .25 * loss_vq_2

            self.add_loss(loss)

            if step > 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_ae', loss_ae, step=step)
                tf.summary.scalar('loss_kl1', loss_vq_1, step=step)
                tf.summary.scalar('loss_kl2', loss_vq_2, step=step)

        return pred, assignment, feat


class GumbelModel(AEModel):
    def __init__(self, conf):
        super().__init__(conf)
        # self.fc = tf.keras.layers.Dense(conf.d_model)

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)
        logits = row_distance(feat, self.context) / self.conf.d_model * -1.
        adj = gumbel_softmax(logits, self.conf.gumbel_temp)
        assignment = gumbel_softmax(logits, self.conf.gumbel_temp, hard=True)
        gumbel_feat = adj @ self.context
        pred = self.decoder(gumbel_feat, training=training)
        if training:
            loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(pred - inputs), axis=1) / 2.)

            loss = loss_ae

            self.add_loss(loss)

            if step > 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_ae', loss_ae, step=step)
        return pred, assignment, feat


class TBHModel(AEModel):
    def __init__(self, conf):
        super().__init__(conf)
        self.fc = tf.keras.layers.Dense(conf.k)
        self.gcn = GCNLayer(conf.d_model)

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)
        logits = self.fc(feat)
        gumbel = gumbel_softmax(logits, self.conf.gumbel_temp)
        assignment = gumbel_softmax(logits, self.conf.gumbel_temp, hard=True)

        _g = tf.nn.l2_normalize(gumbel, axis=1)  # [N K]
        adj = tf.matmul(_g, _g, transpose_b=True)

        _gcn = self.gcn(feat, adj)

        pred = self.decoder(_gcn, training=training)

        if training:
            loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(pred - inputs), axis=1) / 2.)

            loss = loss_ae + tf.reduce_mean(self.context) * 0

            self.add_loss(loss)

            if step > 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_ae', loss_ae, step=step)
                tf.summary.image('adj_hist', tf.reshape(adj, [-1]), step=step)
        return pred, assignment, feat


def step_train(conf, data_1: dict, data_2: dict, model: AEModel, opt: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    feat_2 = data_2['image']

    label_1 = data_1['label']
    label_2 = data_2['label']

    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        agg_1, assign_1, _feat_1 = model(feat_1, step=_step)
        loss = 0
        for l in model.losses:
            loss += l

        gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

    if _step > 0:
        label = label_1  # tf.concat([label_1, label_2], axis=0)
        pred = assign_1  # tf.concat([assign_1, assign_2], axis=0)
        pred = tf.argmax(pred, axis=1)
        feat = _feat_1  # tf.concat([_feat_1, _feat_2], axis=0)

        acc, nmi, ari, sc = hook(feat.numpy(), label.numpy(), pred.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)
        tf.summary.scalar('loss/all', loss, step)

        tf.summary.histogram('eval/assign', pred, step)

    # noinspection PyUnresolvedReferences
    return loss.numpy()
