from __future__ import absolute_import, print_function, division, unicode_literals
import tensorflow as tf


def get_encoder(conf):
    if conf.encoder == 'linear':
        model = tf.keras.Sequential([tf.keras.layers.Dense(conf.d_model * 2),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.Dense(conf.d_model)])
        return model

    if conf.encoder == 'rand_linear':
        model = get_stochastic_linear(conf)
        return model
    if conf.encoder[:5] == 'cifar':
        model = ResNet(BasicBlock, [3, 4, 6, 3], 4, low_dim=128, width=1, k=conf.k, linear=conf.linear)
        return model


def get_stochastic_linear(conf):
    model = tf.keras.Sequential([tf.keras.layers.GaussianNoise(.3),
                                 tf.keras.layers.Dense(conf.d_model * 2),
                                 tf.keras.layers.ReLU(),
                                 tf.keras.layers.GaussianNoise(.3),
                                 tf.keras.layers.Dense(conf.d_model)])
    return model


# noinspection PyAbstractClass
class BasicBlock(tf.keras.layers.Layer):
    EXPANSION = 1

    def __init__(self, channels, filters, strides=1):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same',
                                             use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                             use_bias=False)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.shortcut = tf.keras.Sequential()
        if strides != 1 or channels != (filters * self.EXPANSION):
            self.shortcut.add(tf.keras.layers.Conv2D(filters=self.EXPANSION * filters, kernel_size=1, strides=strides,
                                                     use_bias=False))
            self.shortcut.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=True, mask=None):
        x = tf.nn.relu(self.bn_1(self.conv_1(inputs, training=training), training=training))
        x = self.bn_2(self.conv_2(x, training=training), training=training)
        x += self.shortcut(inputs, training=training)
        return tf.nn.relu(x)


# noinspection PyAbstractClass
class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, pool_len=4, low_dim=128, width=1, k=10, linear=True):
        super().__init__()
        self.channels = 64
        self.pool_len = pool_len
        self.k = k
        self.linear = linear
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.base = int(64 * width)
        self.residual = tf.keras.Sequential([
            self._make_layer(block, self.base, num_blocks[0], stride=1),
            self._make_layer(block, self.base * 2, num_blocks[1], stride=2),
            self._make_layer(block, self.base * 4, num_blocks[2], stride=2),
            self._make_layer(block, self.base * 8, num_blocks[3], stride=2)
        ])
        if self.linear:
            self.fc = tf.keras.layers.Dense(low_dim)
        self.pool = tf.keras.layers.AveragePooling2D(pool_len, pool_len, data_format='channels_last')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.channels, planes, stride))
            self.channels = planes * block.EXPANSION
        return tf.keras.Sequential(layers)

    def call(self, inputs, training=True, mask=None):
        x = tf.nn.relu(self.bn_1(self.conv_1(inputs, training=training), training=training))
        x = self.residual(x, training=training)
        x = self.pool(x)

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1])
        if self.linear:
            x = self.fc(x, training=training)
        return x


def test_resnet():
    model = ResNet(BasicBlock, [3, 4, 6, 3], 4, low_dim=128, width=1)
    a = tf.ones([7, 32, 32, 3])
    b = model(a)
    print(b)


if __name__ == '__main__':
    test_resnet()
