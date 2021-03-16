import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=1024, help='batch_size')
parser.add_argument('-i', '--max_iter', default=200000, help='iter')
parser.add_argument('-d', '--d_model', default=512, help='emb size')
parser.add_argument('-k', '--k', default=10)
parser.add_argument('-t', '--temp', default=0.5)
parser.add_argument('-e', '--encoder', default='linear')
parser.add_argument('-a', '--assigner', default='soft')
parser.add_argument('-g', '--gumbel_temp', default=0.3)
