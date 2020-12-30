import re

import numpy as np
import pandas as pd
from toolz.curried import map, curry
from dateutil.parser import parse

from hhutil.io import read_lines, fmt_path, eglob


@curry
def parse_metric_line(s, p):
    try:
        m = p.search(s)
        return m.group(1), float(m.group(2)), float(m.group(3))
    except AttributeError as e:
        print(p)
        print(s)
        raise e


class LogParser():

    def __init__(self):
        base_pattern = r"""(\d{2}:\d{2}:\d{2}) %s - loss: (\d+\.\d{4}), acc: (\d\.\d{4})"""
        stages = ['train', 'valid']
        self.patterns = {
            k: re.compile(base_pattern % k) for k in stages
        }

    def extract(self, log_file):
        lines = read_lines(log_file)
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

        train_ends, train_losses, train_accs = zip(
            *map(parse_metric_line(p=self.patterns['train']), train_lines))
        valid_ends, valid_losses, valid_accs = zip(
            *map(parse_metric_line(p=self.patterns['valid']), valid_lines))

        train_losses, train_accs, valid_losses, valid_accs = [
            np.array(x) for x in [train_losses, train_accs, valid_losses, valid_accs]]
        train_results = pd.DataFrame({
                "end": train_ends,
                "loss": train_losses,
                "acc": train_accs,
            })
        valid_results = pd.DataFrame({
                "end": valid_ends,
                "loss": valid_losses,
                "acc": valid_accs,
            })
        return train_results, valid_results

    def parse(self, log_file):
        train_ends, train_losses, train_accs, valid_ends, valid_losses, valid_accs = self.extract(log_file)
        total_epochs = len(valid_ends)
        epoch_train_time = (parse(valid_ends[-1]) - parse(valid_ends[0])).seconds / (total_epochs - 1)
        print(f"%.4f(%.4f) %.4f(%.4f) %.1f" % (
            valid_accs[-1], valid_accs.max(),
            train_losses[-1], train_losses[valid_accs.argmax() - len(valid_accs)],
            epoch_train_time))

root = fmt_path("/Users/hrvvi/Code/Library/experiments/CIFAR100-TensorFlow")
parser = LogParser()

def get_logs(n):
    return list(eglob(root / "log", "%d-*.log" % n))

def get_metrics(n, metrics):
    train_res, valid_res = zip(*[parser.extract(log) for log in get_logs(n)])
    results = []
    for m in metrics:
        stage, metric, i = m.split("_")
        res = train_res if stage == 'train' else valid_res
        if i == 'max':
            results.append(np.array([df[metric].max() for df in res]))
        else:
            i = int(i) - 1
            results.append(np.array([df[metric][i] for df in res]))
    return results

exp_ids = [30, 31, 32, 34, 37, 39, 50, 52, 56]
m1, m2 = zip(*[get_metrics(exp_id, ["valid_acc_150", "valid_acc_max"]) for exp_id in exp_ids])

ms = ["valid_acc_%d" % n for n in np.linspace(0, 200, 201)[1:]] + ["valid_acc_max"]
res = list(zip(*[get_metrics(exp_id, ms) for exp_id in exp_ids]))
ref_m = res[-1]
taus = [stats.kendalltau(np.array(m), ref_m).correlation for m in res[:-1]]
