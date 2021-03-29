from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf


class ContrastiveModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.model_1 = 0
        self.model_2 = 0
