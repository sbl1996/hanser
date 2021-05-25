from typing import Optional

import tensorflow as tf

from hanser.distribute.gpu import has_gpu, setup_gpu
from hanser.distribute.tpu import has_tpu, setup_tpu, local_results


def discover_device():
    if has_tpu():
        return 'TPU'
    elif has_gpu():
        return 'GPU'
    else:
        return 'CPU'


def setup_runtime(device='auto', fp16=True):
    if device == 'auto':
        device = discover_device()
    if device == 'TPU':
        setup_tpu(fp16)
    elif device == 'GPU':
        setup_gpu(fp16)
    else: # CPU
        pass


def distribute_datasets(*datasets):
    strategy = tf.distribute.get_strategy()
    if is_distribute_strategy(strategy):
        datasets = [(strategy.experimental_distribute_dataset(ds)
                     if not isinstance(ds, tf.distribute.DistributedDataset) else ds) for ds in datasets]
    datasets = tuple(datasets)
    return datasets


def strategy_run(strategy, fn, args):
    if strategy is not None:
        return strategy.run(fn, args=args)
    else:
        return fn(*args)


def is_tpu_strategy(strategy):
    if strategy is None:
        return False
    return "TPUStrategy" in type(strategy).__name__


def is_distribute_strategy(strategy):
    return is_tpu_strategy(strategy)


def parse_strategy(strategy='auto') -> Optional[tf.distribute.Strategy]:
    if strategy is not None:
        if strategy == 'auto':
            strategy = tf.distribute.get_strategy()
        if not is_distribute_strategy(strategy):
            strategy = None
    return strategy
