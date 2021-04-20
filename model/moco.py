from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.encodec import get_encoder
from layer.gumbel import gumbel_softmax
# from layer.binary_activation import binary_activation
from util.contrastive import moco_loss, loss_with_queue, update_queue
from util.eval import hook
from util.mmc import mmc
from util.dec import softmax_dot_dec


class BaseModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.encoder = get_encoder(conf)
        # self.context = self.add_weight('ContextK', [conf.k, conf.d_model], dtype=tf.float32,
        #                                initializer=tf.initializers.GlorotUniform())

        _context = tf.Variable(mmc(conf.k, conf.d_model), trainable=conf.trainable_context, name='Context',
                               dtype=tf.float32)

        self.context = tf.nn.l2_normalize(_context, axis=1)

    def call(self, inputs, training=True, mask=None):
        x = self.encoder(inputs, training=training)
        return x


class MoCo(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.m = .999
        self.k = conf.k
        self.q = conf.k * conf.q
        self.l = conf.l
        self.temp = conf.temp
        self.sto = conf.sto
        self.gumbel_temp = conf.gumbel_temp
        self.base_1 = BaseModel(conf)
        self.base_2 = BaseModel(conf)
        # self.fc_1 = tf.keras.layers.Dense(conf.d_model)
        _queue_n = tf.Variable(tf.initializers.GlorotUniform()([self.l, conf.d_model]), trainable=False,
                               dtype=tf.float32, name='QueueN')

        self.queue_n = tf.stop_gradient(tf.nn.l2_normalize(_queue_n, axis=1))

        _queue_k = tf.Variable(tf.initializers.GlorotUniform()([self.q, conf.d_model]), trainable=False,
                               dtype=tf.float32, name='QueueK')
        self.queue_k = tf.stop_gradient(tf.nn.l2_normalize(_queue_k, axis=1))

    def call(self, inputs, training=True, mask=None, step=-1):
        x_1 = inputs['image_1']
        x_2 = inputs['image_2']

        feat_1 = tf.nn.l2_normalize(self.base_1(x_1, training=training), axis=1)
        feat_2 = tf.nn.l2_normalize(self.base_2(x_2, training=training), axis=1)

        assign_1, agg_n_1, agg_k_1 = self.cross_rep(feat_1, self.base_1.context, step=step)
        assign_2, agg_n_2, agg_k_2 = self.cross_rep(feat_2, self.base_2.context, step=step)

        if training:
            loss_n = moco_loss(agg_n_1, agg_n_2, self.queue_n, self.temp)
            queue = tf.stop_gradient(self.queue_k + tf.random.normal(tf.shape(self.queue_k), stddev=0.1) + self.queue_k)
            queue = tf.nn.l2_normalize(queue, axis=1)
            loss_k = loss_with_queue(agg_k_1, tf.stop_gradient(agg_k_2), queue, self.k, self.q, self.temp)
            # loss_k_2 = moco_loss(agg_k_1, agg_k_2, self.queue_n, self.temp)

            loss_c = softmax_dot_dec(feat_1, self.base_1.context)

            sim_k = tf.matmul(agg_k_1, agg_k_2, transpose_b=True)
            label_k = tf.eye(self.k)

            loss_k_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label_k, sim_k))

            loss = loss_n + loss_k

            self.add_loss(loss)

            if step >= 0:
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('loss_k', loss_k, step=step)
                tf.summary.scalar('loss_k_2', loss_k_2, step=step)
                tf.summary.scalar('loss_c', loss_c, step=step)
                tf.summary.scalar('loss_n', loss_n, step=step)

        return assign_1, agg_n_2, agg_k_2

    def update_queues(self, new_n, new_k):
        self.queue_n = update_queue(self.queue_n, new_n)
        self.queue_k = update_queue(self.queue_k, new_k)

    def update_momentum(self):
        for i, j in zip(self.base_1.trainable_variables, self.base_2.trainable_variables):
            j.assign(self.m * j + (1 - self.m) * i)

    @property
    def trainable_scope(self):
        return self.base_1.trainable_variables  # + self.fc_1.trainable_variables

    def cross_rep(self, feat, context, step=-1):
        # [N K] agg
        _kn = tf.matmul(context, feat, transpose_b=True)  # [K N]

        stochastic = 1 if step <= self.sto else 0
        assign_n = gumbel_softmax(tf.transpose(_kn), self.gumbel_temp, hard=False, stochastic=stochastic)  # [N K]
        _assign_n = gumbel_softmax(tf.transpose(_kn), self.gumbel_temp, hard=True, stochastic=stochastic)
        # agg_n = assign_n @ context  # [N D]
        # agg_n = tf.nn.l2_normalize(agg_n, axis=1)  # + tf.nn.l2_normalize(self.fc_1(feat), axis=1) * .1
        agg_n = feat  # tf.nn.l2_normalize(feat, axis=1)

        # [K N] agg

        heat_map = tf.transpose(assign_n)
        # eps = tf.random.uniform(tf.shape(heat_map), minval=-0.5, maxval=0.5) * stochastic + 0.5  # [K N]
        # assign_k, _ = binary_activation(heat_map, eps)
        assign_k = heat_map

        def split_agg(_assign, _feat):
            agg_k = _assign @ _feat  # [K D] note that this aggregation is still unnormalized
            normalizer = tf.reduce_sum(_assign, axis=1, keepdims=True) + 1e-8  # [K 1]
            agg_k = agg_k / normalizer
            agg_k = tf.nn.l2_normalize(agg_k, axis=1)  # self.ln_k(agg_k, training=training)
            return agg_k

        agg_k = split_agg(assign_k, feat)
        assignment = tf.argmax(_assign_n, axis=1)
        if step > 0:
            tf.summary.image('adj_nk', assign_n[tf.newaxis, :, :, tf.newaxis] * 255, step=step)
            tf.summary.image('adj_kn', assign_k[tf.newaxis, :, :, tf.newaxis] * 255, step=step)

            tf.summary.histogram('assign_n', assignment, step=step)
            tf.summary.histogram('assign_k', assign_k, step=step)

        return assignment, agg_n, agg_k


def step_train(conf, data: dict, model: MoCo, opt: tf.keras.optimizers.Optimizer, step):
    # feat = data['image']
    label = data['label']
    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        assignment, agg_n, agg_k = model(data, step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_scope)

        opt.apply_gradients(zip(gradients, model.trainable_scope))

    model.update_queues(agg_n, agg_k)
    model.update_momentum()

    if _step > 0:
        acc, nmi, ari, sc = hook(agg_n.numpy(), label.numpy(), assignment.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)
    return loss.numpy()
