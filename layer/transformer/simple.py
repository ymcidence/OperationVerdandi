from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf


class ISAB(tf.keras.layers.Layer):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.d_model = conf.d_model

    # noinspection PyMethodOverriding
    def call(self, feat, context, training=True, mask=None, step=-1):

        return 0
