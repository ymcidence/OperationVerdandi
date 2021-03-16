from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from meta import ROOT_PATH
import typing

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(conf, training=True):
    cases = {
        'cifar100': load_cifar100,
        'cifar_feat': load_cifar_feat
    }
    _data = cases.get(conf.set_name)()

    if training:
        # Note that in clustering, we don't use the conventional train-test split, but instead, have them together
        return _data['train'].concatenate(_data['test']).repeat().shuffle(50000).batch(conf.batch_size).prefetch(
            AUTOTUNE)
    else:
        return _data['test'].batch(conf.batch_size).prefetch(AUTOTUNE)


def load_cifar100() -> typing.Dict[str, tf.data.Dataset]:
    """
    load from tfds
    :return: strictly a dict with keys of 'train' and 'test', with each one at least having 'image' and 'label'
    """
    data_dir = os.path.join(ROOT_PATH, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = tfds.load('cifar100', data_dir=data_dir)
    return data


def load_cifar_feat() -> typing.Dict[str, tf.data.Dataset]:
    """
    reuse TBH tfrecords
    :return: strictly a dict with keys of 'train' and 'test', with each one at least having 'image' and 'label'
    """

    def data_parser(tf_example: tf.train.Example):
        feat_dict = {'id': tf.io.FixedLenFeature([], tf.int64),
                     'feat': tf.io.FixedLenFeature([4096], tf.float32),
                     'label': tf.io.FixedLenFeature([10], tf.float32)}
        features = tf.io.parse_single_example(tf_example, features=feat_dict)

        _id = tf.cast(features['id'], tf.int32)
        _feat = tf.cast(features['feat'], tf.float32)
        _label = tf.cast(tf.argmax(features['label']), tf.int32)
        return {'id': _id,
                'image': _feat,
                'label': _label
                }

    train_name = os.path.join(ROOT_PATH, 'data', 'cifar_feat', 'train.tfrecords')
    test_name = train_name.replace('train.tfrecords', 'test.tfrecords')
    train_data = tf.data.TFRecordDataset(train_name).map(data_parser, num_parallel_calls=50)
    test_data = tf.data.TFRecordDataset(test_name).map(data_parser, num_parallel_calls=50)
    return {'train': train_data,
            'test': test_data}


if __name__ == '__main__':
    load_cifar100()
