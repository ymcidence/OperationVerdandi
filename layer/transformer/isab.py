from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from layer.transformer import attention


class MultiheadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.dff = dff
        self.rate = rate

        self.mha1 = attention.MultiHeadAttention(d_model, num_heads, dense=False)

        self.ffn = tf.keras.layers.Dense(dff, activation=tf.nn.relu)

        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, y, training, mask=None):
        """

        :param x: query [N T1 D1]
        :param y: key and value [N T2 D2]
        :param training:
        :param mask:
        :return:
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # mha has accepts q k v as inputs
        attn1, attn_weights_block1 = self.mha1(x, y, y, mask)  # (batch_size, target_seq_len, d_model)

        out = self.ffn(attn1) + attn1

        return out, attn_weights_block1


class InducedSetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads, dff, rate)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads, dff, rate)

    # noinspection PyMethodOverriding
    def call(self, x, i, training, mask=None, squeeze=True):
        """

        :param x: [* T D]
        :param i: [* K D]
        :param training:
        :param mask:
        :return:
        """
        h, att_kn = self.mab1(i, x, training)  # []
        out, att_nk = self.mab2(x, h, training)
        if squeeze:
            h = tf.squeeze(h)
            out = tf.squeeze(out)
            att_kn = tf.squeeze(att_kn)
            att_nk = tf.squeeze(att_nk)

        return h, out, att_kn, att_nk
