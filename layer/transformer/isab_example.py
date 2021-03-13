from __future__ import absolute_import, print_function, division, unicode_literals

import tensorflow as tf

from layer.transformer.isab import MultiheadAttentionBlock


class ISABExample(tf.keras.layers.Layer):
    def __init__(self, feat_dim, img_dim, txt_dim, context_num=None, **kwargs):
        """

        :param feat_dim: feature dimension
        :param img_dim: original image feature dimension
        :param txt_dim: original text feature dimension
        :param context_num: hyper-parameter, the number of context vectors
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.feat_dim = feat_dim
        if context_num is not None:
            self.context_num = context_num
            self.context_emb = tf.Variable(initial_value=tf.random.normal([context_num, feat_dim], stddev=.01),
                                           trainable=True, dtype=tf.float32, name='ContextEmb')
        self.modal_emb = tf.keras.layers.Embedding(2, feat_dim)
        self.enc_img = tf.keras.layers.Dense(feat_dim)
        self.dec_img = tf.keras.layers.Dense(img_dim)
        self.enc_txt = tf.keras.layers.Dense(feat_dim)
        self.dec_txt = tf.keras.layers.Dense(txt_dim)
        self.mab = MultiheadAttentionBlock(self.feat_dim, 1, self.feat_dim)

    # noinspection PyMethodOverriding
    def call(self, img_feat, txt_feat, training=True, **kwargs):
        batch_size = tf.shape(img_feat)[0]
        zeros = tf.zeros([batch_size], dtype=tf.int32)
        ones = tf.ones([batch_size], dtype=tf.int32)
        modal_emb = self.modal_emb(tf.concat([zeros, ones], axis=0))
        img = self.enc_img(img_feat)
        txt = self.enc_txt(txt_feat)
        data_emb = tf.concat([img, txt], axis=0)
        isab_input = tf.expand_dims(data_emb + modal_emb, axis=0)
        isab_output, affinity = self.mab(isab_input, isab_input)
        affinity = tf.squeeze(affinity)
        z1, z2 = tf.split(tf.squeeze(isab_output), num_or_size_splits=2, axis=1)
        dec_img = self.dec_img(z1)
        dec_txt = self.dec_txt(z2)

        affinity_bias = tf.eye(batch_size)
        affinity_bias = tf.concat([affinity_bias, affinity_bias], axis=0)
        affinity_bias = tf.concat([affinity_bias, affinity_bias], axis=1)
        affinity_mask = tf.ones_like(affinity_bias) - affinity_bias

        affinity = affinity * affinity_mask + affinity_bias

        return dec_img, dec_txt, affinity


if __name__ == '__main__':
    _feat_dim = 512
    _img_dim = 1024  # 例如我们的图片原始特征是1024维
    _txt_dim = 128  # 例如我们的文本原始特征是128维
    model = ISABExample(_feat_dim, _img_dim, _txt_dim)

    _img_feat = tf.ones([64, 1024])  # 假设batch size 64, feature用占位符表示，只用于演示其工作机制
    _txt_feat = tf.ones([64, 128])

    _dec_img, _dec_txt, _affinity = model(_img_feat, _txt_feat, training=True)

    loss = (tf.reduce_mean(tf.square(_img_feat - _dec_img)) + tf.reduce_mean(tf.square(_txt_feat - _dec_txt))) / 2.

    print('this is the additional loss:')
    print(loss)
    print('---------------------------')
    print('this is the similarity matrix we need:')
    print(_affinity)
