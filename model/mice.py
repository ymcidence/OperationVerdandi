from __future__ import absolute_import, division, unicode_literals, print_function

import tensorflow as tf
import numpy as np
from layer.encodec import get_encoder
from argparse import Namespace
from util.mmc import mmc
from util.eval import hook
from util.dec import dec_loss

class BaseModel(tf.keras.Model):
    def __init__(self, conf: Namespace):
        super().__init__()
        self.conf = conf
        self.conf.linear = False
        self.encoder = get_encoder(self.conf)

        self.fc = tf.keras.layers.Dense(conf.k * conf.d_model)

    def call(self, inputs, training=True, mask=None, base_only=False):
        x_base = self.encoder(inputs, training=training)
        if base_only:
            return x_base
        x = self.fc(x_base, training=training)

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.conf.k, -1])
        return tf.nn.l2_normalize(x, axis=-1)  # [N K D]


class GatingModel(tf.keras.Model):
    def __init__(self, conf: Namespace):
        super().__init__()
        self.conf = conf
        self.fc = tf.keras.layers.Dense(conf.d_model)

    def call(self, inputs, training=True, mask=None):
        x = self.fc(inputs, training=training)
        return tf.nn.l2_normalize(x, axis=-1)  # [N D]


class MiCE(tf.keras.Model):

    def __init__(self, conf):
        super().__init__()
        self.m = .999
        self.k = conf.k
        self.l = conf.l
        self.temp = conf.temp
        self.kappa = conf.temp
        self.d_model = conf.d_model

        self.base_1 = BaseModel(conf)
        self.base_2 = BaseModel(conf)
        self.gate = GatingModel(conf)

        self.mu = tf.Variable(tf.initializers.GlorotUniform()([self.k, conf.d_model]), trainable=True,
                              dtype=tf.float32, name='Mu')
        self.queue = tf.Variable(tf.initializers.GlorotUniform()([self.l, self.k, conf.d_model]), trainable=False,
                                 dtype=tf.float32, name='Queue')
        self.omega = tf.Variable(mmc(conf.k, conf.d_model), trainable=False, name='Omega',
                                 dtype=tf.float32)

    def call(self, inputs, training=True, mask=None, step=-1):
        x_f = inputs['image_1']
        x_v = inputs['image_2']
        x_g = inputs['image_3']

        f = self.base_1(x_f, training=training)
        v = self.base_2(x_v, training=training)
        v = tf.stop_gradient(v)

        _g = self.base_1(x_g, training=training, base_only=True)
        g = self.gate(_g, training=training)

        v_f = tf.einsum('nkd,nkd->nk', v, f)
        v_mu = tf.einsum('nkd,kd->nk', v, tf.nn.l2_normalize(self.mu, axis=-1))

        l_pos = v_f + v_mu

        queue_f = tf.einsum('lkd,nkd->nkl', tf.stop_gradient(self.queue), f)
        queue_mu = tf.einsum("lkd,kd->kl", tf.stop_gradient(self.queue), tf.nn.l2_normalize(self.mu, axis=-1))

        l_neg = queue_f + queue_mu[tf.newaxis, :, :]  # [N K L]

        expert_logits = tf.concat([l_pos[:, :, tf.newaxis], l_neg], axis=2) / self.temp
        experts = tf.nn.softmax(expert_logits, axis=2)[:, :, 0]  # [N K]
        gating = tf.nn.softmax(tf.einsum("kd,nd->nk", self.omega, g) / self.kappa, axis=-1)  # [N K]
        variational_q = tf.einsum("nk,nk->nk", gating, experts)
        v_sum = tf.reduce_sum(variational_q, axis=1)[:, tf.newaxis]  # [N 1]
        variational_q = variational_q / v_sum  # [N K]

        assignment = tf.argmax(variational_q, axis=1)

        if training:
            elbo = tf.reduce_sum(
                variational_q * (tf.math.log(gating) + tf.math.log(experts) - tf.math.log(variational_q)), axis=1)
            elbo = tf.reduce_mean(elbo)

            loss = -elbo
            self.add_loss(loss)

            if step >= 0:
                dec = dec_loss(variational_q)
                tf.summary.scalar('dec_loss', dec, step=step)
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.histogram('assign_n', assignment, step=step)

            return assignment, f, v
        else:
            return _g

    def update_queues(self, new_value):
        """

        :param new_value: [N K D]
        :return:
        """
        _queue = tf.concat([new_value, self.queue], axis=0)

        self.queue = tf.stop_gradient(_queue[:self.l, :, :])

    def update_momentum(self):
        for i, j in zip(self.base_1.trainable_variables, self.base_2.trainable_variables):
            j.assign(self.m * j + (1 - self.m) * i)

    def update_initial(self):
        for i, j in zip(self.base_1.variables, self.base_2.variables):
            j.assign(i)

    def update_mu(self, mu_hat):

        _m = tf.convert_to_tensor(mu_hat, dtype=tf.float32)
        self.mu.assign(tf.nn.l2_normalize(_m, axis=-1))

    @property
    def trainable_scope(self):
        return self.base_1.trainable_variables + self.gate.trainable_variables

    def mu_hat(self, assignment, v):

        hard_assignment = tf.one_hot(assignment, self.k, dtype=tf.float32)
        return tf.einsum('nk,nkd->kd', hard_assignment, v)


def step_train(mu_hat, data: dict, model: MiCE, opt: tf.keras.optimizers.Optimizer, step):
    label = data['label']
    _step = -1 if step % 100 > 0 else step

    if step == 0:
        _, _, _ = model(data, step=-1, training=False)
        model.update_initial()

    with tf.GradientTape() as tape:
        assignment, f, v = model(data, step=_step)
        loss = model.losses[0]

        gradients = tape.gradient(loss, model.trainable_scope)

        opt.apply_gradients(zip(gradients, model.trainable_scope))

    model.update_queues(v)
    model.update_momentum()
    mu_hat = mu_hat + model.mu_hat(assignment, v).numpy()

    if _step > 0:
        acc, nmi, ari, sc = hook(f.numpy(), label.numpy(), assignment.numpy())

        tf.summary.scalar('eval/nmi', nmi, step)
        tf.summary.scalar('eval/acc', acc, step)
        tf.summary.scalar('eval/ari', ari, step)
        tf.summary.scalar('eval/sc', sc, step)
    return loss.numpy(), mu_hat


def epoch_train(data: tf.data.Dataset, model: MiCE, opt: tf.keras.optimizers.Optimizer, step_count=0):
    mu_hat = np.zeros([model.k, model.d_model], dtype=np.float)
    this_step = step_count + 0

    for i, d in enumerate(data):
        this_step += 1

        loss, mu_hat = step_train(mu_hat, d, model, opt, this_step)

        if this_step % 50 == 0:
            print('iter {}, loss {}'.format(this_step, loss))

    model.update_mu(mu_hat)

    return this_step
