import argparse
import re

import numpy as np
import pandas as pd
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

    total_epochs = int(epoch_lines[0].split('/')[1])
    train_p = re.compile(f"""({dt_p}) train - loss: ({m_p}), acc: ({m_p})""")
    valid_p = re.compile(f"""({dt_p}) valid - loss: ({m_p}), acc: ({m_p}), acc5: ({m_p})""")
    train_ends, train_losses, train_accs = zip(*map(parse_metric_line(p=train_p), train_lines))
    valid_ends, valid_losses, valid_accs, valid_acc5s = zip(*map(parse_metric_line(p=valid_p), valid_lines))

    train_losses, valid_accs, valid_acc5s = [
        np.array(x) for x in [train_losses, valid_accs, valid_acc5s]]
    return total_epochs, train_start, train_losses, valid_ends, valid_accs, valid_acc5s

def estimate_epoch_cost(valid_ends):
    epoch_seconds = list(map(lambda t: dtime(t[0], t[1]).seconds, zip(valid_ends[1:], valid_ends[:-1])))
    n = len(epoch_seconds)
    second_counts = pd.value_counts(epoch_seconds)
    sum = 0
    tc = 0
    for cost, freq in second_counts.iteritems():
        sum += cost * freq
        tc += freq
        if tc / n > 0.75:
            break
    return sum / tc

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%02d:%02d:%02d' % (hours, minutes, seconds)

parser = argparse.ArgumentParser()
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
args = parser.parse_args()
log_file = args.log

epoch_p = re.compile(r"""Epoch \d+/\d+""")
dt_p = "\d{2}:\d{2}:\d{2}"
m_p = "\d+\.\d{4}|nan"

total_epochs, train_start, train_losses, valid_ends, valid_accs, valid_acc5s = parse_log(log_file)
epoch_cost = estimate_epoch_cost(valid_ends)
total_cost = dtime(valid_ends[0], train_start).seconds + int(epoch_cost * (total_epochs - 1))
total_cost = timedelta(seconds=total_cost)
max_acc_epoch = np.argmax(valid_accs)
print(f"%.2f %.2f %.4f %s %.1f" % (
    valid_accs[max_acc_epoch] * 100, valid_acc5s[max_acc_epoch] * 100, train_losses[-1],
    format_timedelta(total_cost), epoch_cost))