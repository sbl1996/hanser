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
def parse_metric_line(s):
    try:
        end_time = re.search(r"""(\d{2}:\d{2}:\d{2})""", s).group(1)
        metrics = re.findall(r"""([a-z0-9]{2,5}): (\d+\.\d{4})""", s)
        return end_time, metrics
    except AttributeError as e:
        print(s)
        raise e

def parse_metric_lines(lines):
    times, metrics = zip(*map(parse_metric_line, lines))
    assert len(set([len(m) for m in metrics])) == 1
    metric_names = [t[0] for t in metrics[0]]
    d = {
        m: [] for m in metric_names
    }
    for l in metrics:
        for name, val in l:
            d[name].append(float(val))
    d = {
        k: np.array(v) for k, v in d.items()
    }
    d['time'] = times
    return d


def parse_log(fp):
    lines = read_lines(fp)
    train_start = lines[0][:8]
    train_lines = []
    valid_lines = []

    for l in lines:
        if l.startswith("Epoch"):
            continue
        elif ' train ' in l:
            train_lines.append(l)
        elif ' valid ' in l:
            valid_lines.append(l)

    train_metrics = parse_metric_lines(train_lines)
    valid_metrics = parse_metric_lines(valid_lines)

    return train_start, train_metrics, valid_metrics

parser = argparse.ArgumentParser()
parser.add_argument('-k','--key', type=str, required=True)
parser.add_argument('-f','--log', type=str, help='Log file', required=True)
args = parser.parse_args()

log_file = args.log
train_start, train_metrics, valid_metrics = parse_log(log_file)
valid_freq = len(train_metrics['time']) // len(valid_metrics['time'])
total_epochs = len(train_metrics['time'])
epoch_train_time = (parse(valid_metrics['time'][-1]) - parse(valid_metrics['time'][0])).seconds / (total_epochs - valid_freq)
total_cost = timedelta(seconds=dtime(valid_metrics['time'][-1], train_start).seconds)

main_valid_metrics = valid_metrics[args.key] * 100
train_losses = train_metrics['loss']

print(f"%.2f(%.2f) %.4f(%.4f) %s %.1f" % (
    main_valid_metrics[-1], main_valid_metrics.max(),
    train_losses[-1], train_losses[main_valid_metrics.argmax() - len(main_valid_metrics)],
    total_cost, epoch_train_time))