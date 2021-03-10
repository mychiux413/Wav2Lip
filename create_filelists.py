import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--train_ratio', type=float, required=False, default=0.95)
parser.add_argument('--train_limit', type=int, required=False, default=0)
parser.add_argument('--val_limit', type=int, required=False, default=0)

args = parser.parse_args()


def main(args):
    assert os.path.exists(args.data_root)
    assert args.train_ratio < 1.0
    assert args.train_ratio > 0.0

    i_train = 0
    i_val = 0
    train_lines = []
    val_lines = []
    for dirname in os.listdir(args.data_root):
        dirpath = os.path.join(args.data_root, dirname)
        for dataname in os.listdir(dirpath):
            line = os.path.join(dirname, dataname)
            if np.random.rand() < args.train_ratio:
                train_lines.append(line)
            else:
                val_lines.append(line)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)

    with open('filelists/train.txt', 'w') as t:
        for i, line in enumerate(train_lines):
            if args.train_limit > 0 and i > args.train_limit:
                break
            t.write(line + "\n")
            i_train += 1

    with open('filelists/val.txt', 'w') as v:
        for i, line in enumerate(val_lines):
            if args.val_limit > 0 and i > args.val_limit:
                break
            v.write(line + "\n")
            i_val += 1
    print("Create {} train data at: {}".format(i_train, 'filelists/train.txt'))
    print("Create {} val data at: {}".format(i_val, 'filelists/val.txt'))


if __name__ == "__main__":
    main(args)
