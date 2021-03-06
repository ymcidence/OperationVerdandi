from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


def dec_loss(q):
    """

    :param q: [N K]
    :return:
    """
    f_k = tf.reduce_sum(q, axis=0) + 1e-8  # [K]

    q_k = tf.pow(q, 2) / f_k  # [N K]

    norm = tf.reduce_sum(q_k, axis=1)[:, tf.newaxis]  # [N 1]

    p = q_k / norm

    log_p = tf.math.log(p + 1e-8)
    log_q = tf.math.log(q + 1e-8)

    loss = p * (log_p - log_q)  # [N K]

    loss = tf.reduce_sum(loss, axis=1)  # [N]

    return tf.reduce_mean(loss)


def softmax_dot_dec(query, key):
    q = tf.matmul(query, key, transpose_b=True)

    q = tf.nn.softmax(q, axis=1)

    return dec_loss(q)


if __name__ == '__main__':
    a = tf.constant([[.3, .2, .5], [.5, .1, .4]])
    b = dec_loss(a)

    print(b)
