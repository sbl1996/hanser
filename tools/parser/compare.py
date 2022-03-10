import sys
import argparse
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt

from hhutil.io import read_text, fmt_path
from hanser.train.parser import parse_log


def smooth(scalars, weight: float):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def find(f, xs):
    candidates = [x for x in xs if f(x)]
    if len(candidates) == 0:
        return None
    return candidates[0]


def parse(fp):
    text = read_text(fp)
    res = parse_log(text)[1]

    train_metrics = defaultdict(list)
    eval_metrics = defaultdict(list)

    for l in res:
        stages = l.stages
        train_log = find(lambda s: s.name == 'train', stages)
        for k, v in train_log.metrics.items():
            train_metrics[k].append(v)
        eval_log = find(lambda s: s.name == 'valid', stages)
        if eval_log is not None:
            for k, v in eval_log.metrics.items():
                eval_metrics[k].append(v)
    train_metrics = {
        k: np.array(v) for k, v in train_metrics.items()
    }
    eval_metrics = {
        k: np.array(v) for k, v in eval_metrics.items()
    }
    return train_metrics, eval_metrics

parser = argparse.ArgumentParser()
parser.add_argument('-k','--key', type=str, required=True)
parser.add_argument('-f','--logs', nargs='+', help='Log files', required=True)
parser.add_argument('-o','--out', type=str, required=False)
parser.add_argument('-s','--smooth', type=float, required=False, default=0.9)
args = parser.parse_args()

log_files = args.logs
log_files = [fmt_path(f) for f in log_files]
legends = []
for fp in log_files:
    train_metrics, eval_metrics = parse(fp)
    k = args.key
    if k.startswith('train_'):
        metrics = train_metrics
        k = k[6:]
    else:
        if k.startswith('eval_'):
            k = k[5:]
        metrics = eval_metrics
    plt.plot(smooth(metrics[k], args.smooth))
    legends.append(fp.stem)

plt.title(args.key)
plt.legend(legends)
if args.out is not None:
    save_path = fmt_path(args.out) / f"{args.key} {' '.join(legends)}.png"
    plt.savefig(save_path, dpi=500)
else:
    plt.show()