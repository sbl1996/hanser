import argparse

import numpy as np
import pandas as pd

from hhutil.io import read_text
from hanser.train.parser import parse_log


def format_timedelta(total_seconds):
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%02d:%02d:%02d' % (hours, minutes, seconds)


def estimate_epoch_cost(valid_ends):
    epoch_seconds = list(map(lambda t: (t[0] - t[1]).seconds, zip(valid_ends[1:], valid_ends[:-1])))
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


parser = argparse.ArgumentParser()
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
args = parser.parse_args()
log_file = args.log

text = read_text(log_file)
train_start, res = parse_log(text)

total_epochs = res[0].epochs
valid_ends = []
train_losses = []
valid_accs = []
valid_acc5s = []
for l in res:
    train_losses.append(l.stages[0].metrics['loss'])
    if len(l.stages) > 1:
        valid_ends.append(l.stages[1].end)
        metrics = l.stages[1].metrics
        valid_accs.append(metrics['acc'])
        valid_acc5s.append(metrics['acc5'])
train_losses, valid_accs, valid_acc5s = [
    np.array(xs) for xs in [train_losses, valid_accs, valid_acc5s]
]

epoch_cost = estimate_epoch_cost(valid_ends)
total_cost = (valid_ends[0] - train_start).seconds + int(epoch_cost * (total_epochs - 1))
max_acc_epoch = np.argmax(valid_accs)
print(f"%.2f %.2f %.4f %s %.1f" % (
    valid_accs[max_acc_epoch] * 100, valid_acc5s[max_acc_epoch] * 100, train_losses[-1],
    format_timedelta(total_cost), epoch_cost))