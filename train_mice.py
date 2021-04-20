from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf

from time import gmtime, strftime
from meta import ROOT_PATH
from model.mice import MiCE as Model, epoch_train
from util.data.loader import load_data_v2
from util.config import parser
from util.schedule import MoCoSchedule
import json


def main():
    conf = parser.parse_args()

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(ROOT_PATH, 'result', conf.set_name + '_moco')
    task_name = conf.task_name
    save_path = os.path.join(result_path, 'model', task_name + '_' + time_string)
    summary_path = os.path.join(result_path, 'log', task_name + '_' + time_string)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Model(conf)
    data = load_data_v2(conf, training=True, aug=True)

    conf.lr_mode = 'exponential'
    conf.lr_interval = '480,640,800'
    conf.lr_value = .1
    conf.lr = 1
    steps_per_epoch = int(60000 / conf.batch_size)

    lr = MoCoSchedule(conf, steps_per_epoch=steps_per_epoch, initial_epoch=0)
    # opt = tf.keras.optimizers.SGD(lr, momentum=.9)

    opt = tf.keras.optimizers.Adam(.001)

    if conf.restore != '':
        restore_checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)
        restore_checkpoint.restore(conf.restore)
        print('Restored from {}'.format(conf.restore))
        starter = 1
    else:
        starter = 0
    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)

    with writer.as_default():
        for i in range(1000):
            print('This is epoch {}'.format(i))
            starter = epoch_train(data, model, opt, starter)

            save_name = os.path.join(save_path, 'epoch_' + str(i))
            checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    main()
