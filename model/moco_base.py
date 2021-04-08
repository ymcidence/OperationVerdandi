from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf
from layer.encodec import get_encoder
from util.contrastive import moco_loss, update_queue


class MoCoBase(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.m = .999
        self.l = conf.l
        self.temp = conf.temp
        self.base_1 = get_encoder(conf)
        self.base_2 = get_encoder(conf)

        _queue_n = tf.Variable(tf.initializers.GlorotUniform()([self.l, conf.d_model]), trainable=False,
                               dtype=tf.float32, name='QueueN')

        self.queue_n = tf.stop_gradient(tf.nn.l2_normalize(_queue_n))

    def call(self, inputs, training=True, mask=None, step=-1):
        x_1 = inputs['image_1']
        x_2 = inputs['image_2']

        feat_1 = self.base_1(x_1, training=training)
        feat_2 = self.base_2(x_2, training=training)

        feat_1 = tf.nn.l2_normalize(feat_1)
        feat_2 = tf.stop_gradient(tf.nn.l2_normalize(feat_2))

        if training:
            loss = moco_loss(feat_1, feat_2, self.queue_n, self.temp)
            self.add_loss(loss)

            if step >= 0:
                tf.summary.scalar('loss', loss, step=step)

        return feat_1, feat_2

    def update_queues(self, new_n):
        self.queue_n = update_queue(self.queue_n, new_n)

    def update_momentum(self):
        for i, j in zip(self.base_1.trainable_variables, self.base_2.trainable_variables):
            j.assign(self.m * j + (1 - self.m) * i)

    def trainable_scope(self):
        return self.base_1.trainable_variables


def step_train(conf, data: dict, model: MoCoBase, opt: tf.keras.optimizers.Optimizer, step):
    # feat = data['image']
    _step = -1 if step % 100 > 0 else step

    with tf.GradientTape() as tape:
        _, feat_2 = model(data, step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_scope())

        opt.apply_gradients(zip(gradients, model.trainable_scope()))

    model.update_queues(feat_2)
    model.update_momentum()

    return loss.numpy()
