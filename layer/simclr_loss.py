from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf

LARGE_NUM = 1e9

import numpy as np


def cosine_sim(a, b):
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    sim = tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return sim


def simclr_loss(feat_1, feat_2, temp, k=None):
    """

    :param feat_1: [N D]
    :param feat_2: [N D]
    :param temp: temperature
    :param k: if not none, the function only considers the first k rows in the two inputs, otherwise performing
    identical to the original one
    :return:
    """
    sim_11 = cosine_sim(feat_1, feat_1) / temp
    sim_22 = cosine_sim(feat_2, feat_2) / temp
    sim_12 = cosine_sim(feat_1, feat_2) / temp
    sim_21 = cosine_sim(feat_2, feat_1) / temp

    batch_size = tf.shape(feat_1)[0]

    if k is not None:
        mask_11 = tf.eye(k)
        mask_22 = tf.ones([batch_size - k, batch_size - k])
        mask_12 = tf.ones([k, batch_size - k], dtype=tf.float32)
        mask_21 = tf.ones([batch_size - k, k], dtype=tf.float32)

        upper = tf.concat([mask_11, mask_12], axis=1)
        lower = tf.concat([mask_21, mask_22], axis=1)
        mask = tf.concat([upper, lower], axis=0)
        loss_mask_1 = tf.ones(k, dtype=tf.float32)
        loss_mask_2 = tf.zeros(batch_size - k, dtype=tf.float32)
        loss_mask = tf.concat([loss_mask_1, loss_mask_2], axis=0)

    else:
        mask = tf.eye(batch_size)
        loss_mask = tf.ones(batch_size, dtype=tf.float32)
    label = tf.one_hot(tf.range(batch_size), batch_size * 2)

    sim_11 = sim_11 - mask * LARGE_NUM
    sim_22 = sim_22 - mask * LARGE_NUM

    loss_1 = tf.nn.softmax_cross_entropy_with_logits(label, tf.concat([sim_12, sim_11], axis=1))
    loss_2 = tf.nn.softmax_cross_entropy_with_logits(label, tf.concat([sim_21, sim_22], axis=1))

    # loss = tf.reduce_mean(loss_1 + loss_2)
    loss = tf.reduce_sum((loss_1 + loss_2) * loss_mask) / tf.reduce_sum(loss_mask)
    return loss, sim_12, label
