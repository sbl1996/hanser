import os

import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
import tensorflow.keras.mixed_precision.experimental as mixed_precision


def setup(datasets, fp16=True, device='auto'):
    if device == 'auto':
        strategy = get_colab_tpu()
        if strategy:
            device = 'TPU'
        elif tf.config.list_physical_devices('GPU'):
            device = 'GPU'
        else:
            device = 'CPU'
    elif device == 'TPU':
        strategy = get_colab_tpu()

    if device == 'TPU':
        if fp16:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_policy(policy)
        tf.distribute.experimental_set_strategy(strategy)
        return [strategy.experimental_distribute_dataset(ds) for ds in datasets]
    elif device == 'GPU':
        if fp16:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        return datasets
    else:
        return datasets


def get_colab_tpu():
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    if tpu_address:
        tpu_address = "grpc://" + tpu_address
        tf.keras.backend.clear_session()
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        return strategy


def auth():
    from google.colab import auth
    auth.authenticate_user()


def local_results(strategy, values):
    if isinstance(values, PerReplica):
        return strategy.experimental_local_results(values)
    elif isinstance(values, (list, tuple)):
        return values.__class__(local_results(strategy, v) for v in values)
    elif isinstance(values, dict):
        return { k: local_results(strategy, v) for k, v in values.items() }
    else:
        return values