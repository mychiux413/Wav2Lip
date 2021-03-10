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
    is_train_limit = False
    is_val_limit = False
    with open('filelists/train.txt', 'w') as t, open('filelists/val.txt', 'w') as v:
        for dirname in os.listdir(args.data_root):
            dirpath = os.path.join(args.data_root, dirname)
            for dataname in os.listdir(dirpath):
                if args.train_limit > 0 and i_train > args.train_limit:
                    is_train_limit = True
                if args.val_limit > 0 and i_val > args.val_limit:
                    is_val_limit = True
                line = os.path.join(dirname, dataname)
                if np.random.rand() < args.train_ratio and not is_train_limit:
                    t.write(line + "\n")
                    i_train += 1
                elif not is_val_limit:
                    v.write(line + "\n")
                    i_val += 1
    print("Create {} train data at: {}".format(i_train, 'filelists/train.txt'))
    print("Create {} val data at: {}".format(i_val, 'filelists/val.txt'))

if __name__ == "__main__":
    main(args)
