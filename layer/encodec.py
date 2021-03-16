from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf


def get_encoder(conf):
    if conf.encoder == 'linear':
        model = tf.keras.Sequential([tf.keras.layers.Dense(conf.d_model * 2),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.Dense(conf.d_model)])
        return model
