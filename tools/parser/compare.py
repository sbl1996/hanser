import sys
import numpy as np

import matplotlib.pyplot as plt

from hhutil.io import read_text
from hanser.train.parser import parse_log


def smooth(scalars, weight: float):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def parse(fp):
    text = read_text(fp)
    res = parse_log(text)[1]

    train_losses = []
    valid_accs = []
    valid_acc5s = []
    for l in res:
        train_losses.append(l.stages[0].metrics['loss'])
        if len(l.stages) > 1:
            metrics = l.stages[1].metrics
            valid_accs.append(metrics['acc'])
            valid_acc5s.append(metrics['acc5'])
    train_losses, valid_accs, valid_acc5s = [
        np.array(xs) for xs in [train_losses, valid_accs, valid_acc5s]
    ]
    return train_losses, valid_accs, valid_acc5s

log_files = sys.argv[1:]

for fp in log_files:
    train_losses, valid_accs, valid_acc5s = parse(fp)
    plt.plot(smooth(valid_accs, 0.9))
plt.legend(log_files)
plt.savefig("/Users/hrvvi/Downloads/res_0_9.png", dpi=500)