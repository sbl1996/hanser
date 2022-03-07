import os

import tensorflow as tf

def setup_tpu(fp16=True, connect=None):
    assert has_tpu()
    tf.keras.backend.clear_session()

    tpu_address = get_tpu_address()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
    if tpu_address != 'local' or connect:
        tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)
    tf.distribute.experimental_set_strategy(strategy)

    if fp16:
        from packaging.version import parse as vparse
        if vparse(tf.__version__) >= vparse("2.4"):
            import tensorflow.keras.mixed_precision as mixed_precision
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
        else:
            import tensorflow.keras.mixed_precision.experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_policy(policy)


def has_tpu():
    return get_tpu_address() is not None


def get_tpu_address():
    return os.environ.get("TPU_NAME")


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