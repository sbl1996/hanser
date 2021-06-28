import argparse

import numpy as np

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

train_ends = []
train_losses = []
main_valid_metrics = []
for l in res:
    train_ends.append(l.stages[0].end)
    train_losses.append(l.stages[0].metrics['loss'])
    main_valid_metrics.append(l.stages[1].metrics[args.key])
train_losses = np.array(train_losses)
main_valid_metrics = np.array(main_valid_metrics)

main_valid_metrics = main_valid_metrics * 100
final_metric = main_valid_metrics[-1]
final_loss = train_losses[-1]
max_metric = np.max(main_valid_metrics)
max_metric_loss = train_losses[np.argmax(main_valid_metrics) - len(main_valid_metrics)]
total_epochs = len(train_losses)
epoch_time = ((train_ends[-1] - train_ends[0]).seconds / (total_epochs - 1))

print(f"%.2f(%.2f) %.4f(%.4f) %.1f" % (
    final_metric, max_metric, final_loss, max_metric_loss, epoch_time))
