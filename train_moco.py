from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf

from time import gmtime, strftime
from meta import ROOT_PATH
from model.moco import MoCo as Model, step_train
from util.data.loader import load_data
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
    data = load_data(conf, training=True, aug=True)
    data_iter = iter(data)

    conf.lr_mode = 'exponential'
    conf.lr_interval = '120,160'
    conf.lr_value = .95
    conf.lr = 0.001
    steps_per_epoch = int(60000 / conf.batch_size)

    lr = MoCoSchedule(conf, steps_per_epoch=steps_per_epoch, initial_epoch=0)
    # opt = tf.keras.optimizers.SGD(lr, momentum=.9)

    opt = tf.keras.optimizers.Adam(lr)

    if conf.restore != '':
        restore_checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)
        restore_checkpoint.restore(conf.restore)
        print('Restored from {}'.format(conf.restore))
    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(actor_opt=opt, model=model)

    with open(os.path.join(save_path, 'conf.txt'), 'w') as f:
        json.dump(conf.__dict__, f, indent=2)

    for i in range(conf.max_iter):
        batch_1 = next(data_iter)
        with writer.as_default():

            loss = step_train(conf, batch_1, model, opt, i)

            if i == 0:
                print(model.summary())

            if i % 50 == 0:
                print('iter {}, loss {}'.format(i, loss))

            if i % 5000 == 0 and i > 0:
                save_name = os.path.join(save_path, '_' + str(i))
                checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    main()
