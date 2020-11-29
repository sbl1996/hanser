import argparse
import re

import numpy as np
from toolz.curried import map, curry
from dateutil.parser import parse

from hhutil.io import read_lines

@curry
def parse_metric_line(s, p):
    try:
        m = p.search(s)
        return m.group(1), float(m.group(2)), float(m.group(3))
    except AttributeError as e:
        print(s)
        raise e


def parse_log(fp):
    lines = read_lines(fp)
    epoch_lines = []
    train_lines = []
    valid_lines = []

    for l in lines:
        if l.startswith("Epoch"):
            epoch_lines.append(l)
        elif ' train ' in l:
            train_lines.append(l)
        elif ' valid ' in l:
            valid_lines.append(l)

    train_ends, train_losses, train_accs = zip(*map(parse_metric_line(p=train_p), train_lines))
    valid_ends, valid_losses, valid_accs = zip(*map(parse_metric_line(p=valid_p), valid_lines))

    train_losses, train_accs, valid_losses, valid_accs = [
        np.array(x) for x in [train_losses, train_accs, valid_losses, valid_accs]]
    return train_ends, train_losses, train_accs, valid_ends, valid_losses, valid_accs

parser = argparse.ArgumentParser()
parser.add_argument('-k','--keys', nargs='+', help='Keys', default=['loss', 'acc'])
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
args = parser.parse_args()
assert len(args.keys) == 2
k1, k2 = args.keys

epoch_p = re.compile(r"""Epoch \d+/\d+""")
train_p = re.compile(r"""(\d{2}:\d{2}:\d{2}) train - %s: (\d+\.\d{4}), %s: (\d\.\d{4})""" % (k1, k2))
valid_p = re.compile(r"""(\d{2}:\d{2}:\d{2}) valid - %s: (\d+\.\d{4}), %s: (\d\.\d{4})""" % (k1, k2))

log_file = args.log
train_ends, train_losses, train_accs, valid_ends, valid_losses, valid_accs = parse_log(log_file)
total_epochs=  len(valid_ends)
epoch_train_time = (parse(valid_ends[-1]) - parse(valid_ends[0])).seconds / (total_epochs - 1)
print(f"%.4f(%.4f) %.4f(%.4f) %.1f" % (
    valid_accs[-1], valid_accs.max(),
    train_losses[-1], train_losses[valid_accs.argmax()-len(valid_accs)],
    epoch_train_time))