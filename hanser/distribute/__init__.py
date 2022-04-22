from typing import Optional

import tensorflow as tf

from hanser.distribute.gpu import has_gpu, setup_gpu
from hanser.distribute.tpu import has_tpu, setup_tpu, local_results


__all__ = ["setup_runtime", "distribute_datasets", "parse_strategy", "strategy_run", "is_distribute_strategy", "local_results", "reduce_per_replica"]


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


def _is_per_replica_instance(obj):
    return (isinstance(obj, tf.distribute.DistributedValues) and
            isinstance(obj, tf.__internal__.CompositeTensor))


def reduce_per_replica(values, strategy, reduction='first'):
    def _reduce(v):
        if not _is_per_replica_instance(v):
            return v
        elif reduction == 'first':
            return strategy.experimental_local_results(v)[0]
        elif reduction == 'concat':
            return tf.concat(strategy.experimental_local_results(v), axis=0)
        else:
            raise ValueError('`reduction` must be "first" or "concat". Received: '
                             f'reduction={reduction}.')

    return tf.nest.map_structure(_reduce, values)