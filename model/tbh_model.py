from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# from layer.functional import row_distance, vq
from layer.gcn import GCNLayer
from layer.gumbel import gumbel_softmax
from layer.encodec import get_encoder
from util.eval import hook


class TBHModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.fc = tf.keras.layers.Dense(conf.k)
        self.gcn = GCNLayer(conf.d_model)

        self.encoder = get_encoder(conf)

        self.k = conf.k
        self.conf = conf
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2048),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(4096)
        ])

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)
        logits = self.fc(feat)
        gumbel = gumbel_softmax(logits, self.conf.gumbel_temp)
        assignment = gumbel_softmax(logits, self.conf.gumbel_temp, hard=True)

        _g = tf.nn.l2_normalize(gumbel, axis=1)  # [N K]
        adj = tf.matmul(_g, _g, transpose_b=True)
        adj = tf.pow(tf.nn.relu(adj), 1)

        _gcn = tf.nn.sigmoid(self.gcn(feat, adj))


        pred = self.decoder(_gcn, training=training)

        if training:
            loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(pred - inputs), axis=1) / 2.)

            loss = loss_ae

            self.add_loss(loss)
            n = tf.cast(tf.shape(feat)[0], tf.float32)

            if step > 0:
                tf.summary.scalar('loss_ae', loss_ae, step=step)
                tf.summary.image('adj', adj[tf.newaxis, :, :, tf.newaxis], step=step)
                tf.summary.histogram('adj_hist', tf.reshape(adj, [-1]), step=step)
                tf.summary.scalar('adj_sum', tf.reduce_sum(adj) / n, step=step)
        return gumbel, assignment, feat


class AdvModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.model = TBHModel(conf)
        self.dis = tf.keras.layers.Dense(1, activation='sigmoid')
        self.conf = conf

    def call(self, inputs, training=True, mask=None, step=-1):
        gumbel, assignment, feat = self.model(inputs, training=training, step=step)

        sample = tf.ones_like(gumbel, dtype=tf.float32) / self.conf.k  # [N K]
        sample = tf.random.categorical(tf.math.log(sample), 1, dtype=tf.int32)
        sample = tf.one_hot(tf.squeeze(sample), self.conf.k, dtype=tf.float32)
        dis = self.dis(gumbel)
        dis_sample = self.dis(sample)

        if training:
            loss_ae = self.model.losses[0]
            actor_loss = loss_ae - self.adv_loss(dis_sample, dis)
            critic_loss = self.adv_loss(dis_sample, dis) * 10
            self.add_loss([actor_loss, critic_loss])

            if step > 0:
                tf.summary.scalar('actor', actor_loss, step=step)
                tf.summary.scalar('critic', critic_loss, step=step)

        return gumbel, assignment, feat

    @staticmethod
    @tf.function
    def adv_loss(real, fake):
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real))
        fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake))
        total_loss = real_loss + fake_loss
        return total_loss

    @staticmethod
    @tf.function
    def reconstruction_loss(pred, origin):
        return tf.reduce_mean(tf.nn.l2_loss(pred - origin))


def step_train(conf, data_1: dict, data_2: dict, model: AdvModel, opt_1: tf.keras.optimizers.Optimizer,
               opt_2: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    # feat_2 = data_2['image']

    label_1 = data_1['label']
    # label_2 = data_2['label']

    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        agg_1, assign_1, _feat_1 = model(feat_1, step=_step)
        actor_loss = model.losses[0]
        critic_loss = model.losses[1]
        actor_scope = model.model.trainable_variables
        critic_scope = model.dis.trainable_variables

        actor_gradients = actor_tape.gradient(actor_loss, actor_scope)
        critic_gradients = critic_tape.gradient(critic_loss, critic_scope)

        opt_1.apply_gradients(zip(actor_gradients, actor_scope))
        opt_2.apply_gradients(zip(critic_gradients, critic_scope))

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
        # tf.summary.scalar('loss/all', loss, step)

        tf.summary.histogram('eval/assign', pred, step)

    # noinspection PyUnresolvedReferences
    return actor_loss.numpy()
