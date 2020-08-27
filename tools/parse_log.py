import re

import numpy as np
from helm.io import read_lines, eglob
from toolz.curried import map


def lmap(f, *iterables):
    return list(map(f, *iterables))


epoch_p = re.compile(r"""Epoch \d+/\d+""")
train_p = re.compile(r""".* Train \d+/\d+ - loss: (\d\.\d{4}) - acc: (\d\.\d{4})""")
valid_p = re.compile(r""".* Valid \d+/\d+ - loss: (\d\.\d{4}) - acc: (\d\.\d{4})""")


def parse_train(s):
    m = train_p.search(s)
    return float(m.group(1)), float(m.group(2))

def parse_valid(s):
    m = valid_p.search(s)
    return float(m.group(1)), float(m.group(2))

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
        elif 'Train' in l:
            train_lines.append(l)
        elif 'Valid' in l:
            valid_lines.append(l)

    train_losses, train_accs = zip(*map(parse_train, train_lines))
    valid_losses, valid_accs = zip(*map(parse_valid, valid_lines))

    train_losses, train_accs, valid_losses, valid_accs = [
        np.array(x) for x in [train_losses, train_accs, valid_losses, valid_accs]]
    return train_losses, train_accs, valid_losses, valid_accs


log_file = "/Users/hrvvi/Code/Library/experiments/CIFAR10-DARTS/85-1.log"
train_losses, train_accs, valid_losses, valid_accs = parse(log_file)
print(valid_accs[-1], valid_accs.max(), train_losses[-1])