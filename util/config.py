import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=384, type=int, help='batch_size')
parser.add_argument('-i', '--max_iter', default=400000, help='iter')
parser.add_argument('-d', '--d_model', default=128, help='emb size')
parser.add_argument('-k', '--k', default=10)
parser.add_argument('-t', '--temp', default=1, type=float)
parser.add_argument('-e', '--encoder', default='cifar')
parser.add_argument('-a', '--assigner', default='soft')
parser.add_argument('-g', '--gumbel_temp', default=0.3, type=float,
                    help='temp of the gumbel trick. Set it <0 to disable stochasticity')
parser.add_argument('-n', '--set_name', default='cifar10', type=str, help='dataset')
parser.add_argument('-tn', '--task_name', default='default', help='task name')
parser.add_argument('-sh', '--shuffle', default=60000)
parser.add_argument('-q', '--q', default=20, type=int)
parser.add_argument('-l', '--l', default=16384, type=int)
parser.add_argument('-st', '--sto', default=100000, type=int)
parser.add_argument('-r', '--restore', default='', type=str)
parser.add_argument('-tc', '--trainable_context', default=True, type=bool)
