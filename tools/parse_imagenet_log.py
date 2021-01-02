import argparse
import re

import numpy as np
from toolz.curried import map, curry
from datetime import timedelta
from dateutil.parser import parse

from hhutil.io import read_lines

def dtime(end, start):
    return parse(end) - parse(start)

@curry
def parse_metric_line(s, p):
    try:
        m = p.search(s)
        return [m.group(1)] + [float(g) for g in m.groups()[1:]]
    except AttributeError as e:
        print(s)
        raise e


def parse_log(fp):
    lines = read_lines(fp)
    train_start = lines[0][:8]
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

    train_p = re.compile(f"""({dt_p}) train - loss: ({m_p}), acc: ({m_p})""")
    valid_p = re.compile(f"""({dt_p}) valid - loss: ({m_p}), acc: ({m_p}), acc5: ({m_p})""")
    train_ends, train_losses, train_accs = zip(*map(parse_metric_line(p=train_p), train_lines))
    valid_ends, valid_losses, valid_accs, valid_acc5s = zip(*map(parse_metric_line(p=valid_p), valid_lines))

    train_losses, valid_accs, valid_acc5s = [
        np.array(x) for x in [train_losses, valid_accs, valid_acc5s]]
    return train_start, train_losses, valid_ends, valid_accs, valid_acc5s

parser = argparse.ArgumentParser()
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
args = parser.parse_args()

epoch_p = re.compile(r"""Epoch \d+/\d+""")
dt_p = "\d{2}:\d{2}:\d{2}"
m_p = "\d+\.\d{4}"

log_file = args.log
train_start, train_losses, valid_ends, valid_accs,valid_acc5s = parse_log(log_file)
total_time = timedelta(seconds=dtime(valid_ends[-1], train_start).seconds)
total_epochs = len(valid_ends)
epoch_train_time = (parse(valid_ends[-1]) - parse(valid_ends[2])).seconds / (total_epochs - 3)
print(f"%.2f %.2f %.4f %s %.1f" % (
    valid_accs[-1] * 100, valid_acc5s[-1] * 100, train_losses[-1],
    total_time, epoch_train_time))