import sys
import re

import numpy as np
from toolz.curried import map

from hhutil.io import read_lines

epoch_p = re.compile(r"""Epoch \d+/\d+""")
train_p = re.compile(r""".* train - loss: (\d+\.\d{4}), acc: (\d\.\d{4})""")
valid_p = re.compile(r""".* valid - loss: (\d+\.\d{4}), acc: (\d\.\d{4})""")


def parse_train(s):
    try:
        m = train_p.search(s)
        return float(m.group(1)), float(m.group(2))
    except AttributeError as e:
        print(s)
        raise e

def parse_valid(s):
    try:
        m = valid_p.search(s)
        return float(m.group(1)), float(m.group(2))
    except Exception as e:
        print(s)
        raise e


def parse(fp):
    lines = read_lines(fp)
    epoch_lines = []
    train_lines = []
    valid_lines = []

    i = 0
    while not lines[i].startswith("Epoch 1/"):
        i += 1

    lines = lines[i + 1:]
    for l in lines:
        if 'Epoch' in l:
            epoch_lines.append(l)
        elif 'train' in l:
            train_lines.append(l)
        elif 'valid' in l:
            valid_lines.append(l)

    train_losses, train_accs = zip(*map(parse_train, train_lines))
    valid_losses, valid_accs = zip(*map(parse_valid, valid_lines))

    train_losses, train_accs, valid_losses, valid_accs = [
        np.array(x) for x in [train_losses, train_accs, valid_losses, valid_accs]]
    return train_losses, train_accs, valid_losses, valid_accs

log_file = sys.argv[1]
train_losses, train_accs, valid_losses, valid_accs = parse(log_file)
print(f"%.4f(%.4f) %.4f(%.4f)" % (valid_accs[-1], valid_accs.max(),
                                  train_losses[-1], train_losses[valid_accs.argmax()-len(valid_accs)]))
