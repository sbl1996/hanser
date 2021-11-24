import os

import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision


def setup_tpu(fp16=True):
    assert has_tpu()
    tf.keras.backend.clear_session()

    tpu_address = get_colab_tpu_address()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)
    tf.distribute.experimental_set_strategy(strategy)

    if fp16:
        policy = mixed_precision.Policy('mixed_bfloat16')
        mixed_precision.set_global_policy(policy)


def has_tpu():
    return get_colab_tpu_address() is not None


def get_colab_tpu_address():
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    if tpu_address is not None:
        tpu_address = "grpc://" + tpu_address
    return tpu_address


def is_tpu_strategy(strategy):
    if strategy is None:
        return False
    return "TPUStrategy" in type(strategy).__name__


def local_results(values, strategy='auto'):
    if strategy == 'auto':
        strategy = tf.distribute.get_strategy()
    def func(x):
        if "PerReplica" in type(x).__name__:
            assert is_tpu_strategy(strategy)
            x = strategy.experimental_local_results(x)
            return tf.concat(x, axis=0)
        return x
    return tf.nest.map_structure(func, values)