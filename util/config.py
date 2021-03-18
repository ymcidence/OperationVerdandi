import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch_size')
parser.add_argument('-i', '--max_iter', default=200000, help='iter')
parser.add_argument('-d', '--d_model', default=512, help='emb size')
parser.add_argument('-k', '--k', default=10)
parser.add_argument('-t', '--temp', default=1)
parser.add_argument('-e', '--encoder', default='linear')
parser.add_argument('-a', '--assigner', default='soft')
parser.add_argument('-g', '--gumbel_temp', default=0.3, type=float,
                    help='temp of the gumbel trick. Set it <0 to disable stochasticity')
parser.add_argument('-n', '--set_name', default='cifar_feat', help='dataset')
parser.add_argument('-tn', '--task_name', default='default', help='task name')
parser.add_argument('-sh', '--shuffle', default=60000)
