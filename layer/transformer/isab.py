from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from layer.transformer import attention


class MultiheadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = attention.MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.layers.Dense(dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    # noinspection PyMethodOverriding
    def call(self, x, y, training, mask=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, y, y, mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, attn_weights_block1


class InducedSetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d_model, num_heads, dff, rate)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads, dff, rate)

    # noinspection PyMethodOverriding
    def call(self, x, i, training, mask=None):
        h, att = self.mab1(i, x, training)
        out, _ = self.mab2(x, h, training)

        return out
