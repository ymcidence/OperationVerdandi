from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf
from layer.gumbel import gumbel_softmax
from layer.encodec import get_encoder
from layer.binary_activation import binary_activation
# from layer.simclr_loss import simclr_loss
from util.eval import hook
from util.contrastive import loss_with_queue, update_queue, simclr_loss


class SimpleModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.gumbel_temp = conf.gumbel_temp
        self.q = conf.k * conf.q
        self.k = conf.k

        self.context = tf.Variable(tf.initializers.GlorotNormal()([conf.k, conf.d_model]), trainable=True,
                                   dtype=tf.float32, name='ContextEmb')
        self.encoder = get_encoder(conf)
        self.ln_n = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.ln_k = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation=tf.nn.relu),
            tf.keras.layers.Dense(4096)
        ])
        queue = tf.Variable(tf.initializers.GlorotUniform()([self.q, conf.d_model]), trainable=False,
                            dtype=tf.float32, name='Queue')

        self.queue = tf.nn.l2_normalize(queue)

    def call(self, inputs, training=True, mask=None, step=-1):
        feat = self.encoder(inputs, training=training)

        # [N K] agg
        _kn = tf.matmul(self.context, feat, transpose_b=True)  # [K N]
        assign_n = gumbel_softmax(tf.transpose(_kn), self.gumbel_temp, hard=False)  # [N K]
        _assign_n = gumbel_softmax(tf.transpose(_kn), self.gumbel_temp, hard=True)
        agg_n = assign_n @ self.context  # [N D]
        agg_n = self.ln_n(agg_n, training=training)
        pred = self.decoder(agg_n, training=training)

        # [K N] agg
        heat_map = _kn
        eps = tf.random.uniform(tf.shape(heat_map), minval=0, maxval=.5)  # [K N]
        assign_k, _ = binary_activation(heat_map, eps)  # [K N]

        def split_agg(_assign, _feat):
            agg_k = _assign @ _feat  # [K D] note that this aggregation is still unnormalized
            normalizer = tf.reduce_mean(_assign, axis=1, keepdims=True) + 1e-8  # [K 1]
            agg_k = agg_k / normalizer
            agg_k = tf.nn.l2_normalize(agg_k)  # self.ln_k(agg_k, training=training)
            return agg_k

        assign_k_1, assign_k_2 = tf.split(assign_k, 2, axis=1)
        feat_1, feat_2 = tf.split(feat, 2, axis=0)
        agg_k_1 = split_agg(assign_k_1, feat_1)
        agg_k_2 = split_agg(assign_k_2, feat_2)

        assignment = tf.argmax(_assign_n, axis=1)
        if training:
            loss_clr_0, _, _ = simclr_loss(agg_k_1, agg_k_2, temp=self.conf.temp)
            queue = tf.stop_gradient(self.queue + tf.random.normal(tf.shape(self.queue), stddev=0.1) + self.queue)
            queue = tf.nn.l2_normalize(queue)
            loss_clr_1 = loss_with_queue(agg_k_1, tf.stop_gradient(agg_k_2), queue, self.k, self.q, self.conf.temp)
            queue = tf.stop_gradient(self.queue + tf.random.normal(tf.shape(self.queue), stddev=0.1) + self.queue)
            queue = tf.nn.l2_normalize(queue)
            loss_clr_2 = loss_with_queue(agg_k_2, tf.stop_gradient(agg_k_1), queue, self.k, self.q, self.conf.temp)
            loss_clr = loss_clr_1 + loss_clr_2 + loss_clr_0

            loss_ae = tf.reduce_mean(tf.square(pred - inputs)) / 2.
            loss = loss_clr + loss_ae

            self.add_loss(loss)

            if step > 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_ae', loss_ae, step=step)
                tf.summary.scalar('loss_clr', loss_clr, step=step)

                tf.summary.histogram('assign_n', assignment, step=step)
                tf.summary.histogram('assign_k', assign_k, step=step)

                tf.summary.image('adj_nk', assign_n[tf.newaxis, :, :, tf.newaxis] * 255, step=step)
                tf.summary.image('adj_kn', assign_k[tf.newaxis, :, :, tf.newaxis] * 255, step=step)

        return assignment, feat, tf.concat([agg_k_1, agg_k_2], axis=0)


def step_train(conf, data_1: dict, data_2: dict, model: SimpleModel, opt: tf.keras.optimizers.Optimizer, step):
    feat_1 = data_1['image']
    feat_2 = data_2['image']

    label_1 = data_1['label']
    label_2 = data_2['label']

    feat_1 = tf.concat([feat_1, feat_2], axis=0)
    label_1 = tf.concat([label_1, label_2], axis=0)

    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        assign_nk, _feat_1, agg_k = model(feat_1, step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

    model.queue = update_queue(model.queue, agg_k)

    if _step > 0:
        label = label_1
        pred = assign_nk
        feat = _feat_1

        acc, nmi, ari, sc = hook(feat.numpy(), label.numpy(), pred.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)
    return loss.numpy()
