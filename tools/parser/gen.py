import argparse

import numpy as np
from datetime import timedelta

from hanser.train.parser import parse_log
from hhutil.io import read_text

parser = argparse.ArgumentParser()
parser.add_argument('-k','--key', default='acc', type=str)
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
parser.add_argument('--mode', choices=["final", "max", "all"], default='all')
args = parser.parse_args()

log_file = args.log
text = read_text(log_file)
train_start, res = parse_log(text)

def estimate_epoch_train_time(train_ends, valid_ends):
    times = []
    for train_end, valid_end in zip(train_ends[1:], valid_ends[:-1]):
        times.append(train_end - valid_end.seconds)
    return float(np.mean(times))


log_file = args.log
train_start, res = parse_log(log_file)

train_ends = []
valid_ends = []
train_losses = []
main_valid_metrics = []
for l in res:
    train_ends.append(l.stages[0].end)
    train_losses.append(l.stages[0].metrics['loss'])
    valid_ends.append(l.stages[1].end)
    main_valid_metrics.append(l.stages[1].metrics[args.key])
train_losses = np.array(train_losses)
main_valid_metrics = np.array(main_valid_metrics)


epoch_train_time = estimate_epoch_train_time(train_metrics['time'], valid_metrics['time'])
total_cost = timedelta(seconds=dtime(valid_metrics['time'][-1], train_start).seconds)
main_valid_metrics = valid_metrics[args.key] * 100
train_losses = train_metrics['loss']

final_metric = main_valid_metrics[-1]
final_loss = train_losses[-1]
max_metric = main_valid_metrics.max()
max_metric_loss = train_losses[main_valid_metrics.argmax() - len(main_valid_metrics)]
total_epochs = len(train_losses)
epoch_time = (dtime(train_metrics['time'][-1], train_metrics['time'][0]).seconds / (total_epochs - 1))

mode = args.mode
if mode == 'final':
    print(f"%.2f %.4f %s %.1f %.1f" % (
        final_metric, final_loss, total_cost, epoch_time, epoch_train_time))
elif mode == 'max':
    print(f"%.2f %.4f %s %.1f %.1f" % (
        max_metric, max_metric_loss, total_cost, epoch_time, epoch_train_time))
else:
    print(f"%.2f(%.2f) %.4f(%.4f) %s %.1f %.1f" % (
        final_metric, max_metric, final_loss, max_metric_loss, total_cost, epoch_time, epoch_train_time))
