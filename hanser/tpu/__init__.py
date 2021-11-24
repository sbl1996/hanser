import os
import warnings
from typing import Optional

import multiprocessing

import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision


def setup(datasets, fp16=True, device='auto', cross_device_ops=None):
    warnings.warn(
        "setup will be deprecated in hanser 1.0, "
        "use setup_runtime and distribute_datasets from hanser.distribute instead.",
        DeprecationWarning,
    )

    if device == 'auto':
        strategy = get_colab_tpu()
        if strategy:
            device = 'TPU'
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) == 0:
                device = 'CPU'
            elif len(gpus) == 1:
                device = 'GPU'
            else:
                device = 'GPUs'
                strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
                set_gpu_thread_mode_and_count(len(gpus))
    elif device == 'TPU':
        strategy = get_colab_tpu()
    elif isinstance(device, list):
        strategy = tf.distribute.MirroredStrategy(devices=device, cross_device_ops=cross_device_ops)
        set_gpu_thread_mode_and_count(len(device))
    else:
        strategy = None

    if device == 'TPU':
        if fp16:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
        tf.distribute.experimental_set_strategy(strategy)
        return [
            (strategy.experimental_distribute_dataset(ds)
             if not isinstance(ds, tf.distribute.DistributedDataset) else ds)
            for ds in datasets]
    elif device == 'GPU':
        if fp16:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        return datasets
    elif isinstance(device, list) or device == 'GPUs':
        tf.distribute.experimental_set_strategy(strategy)
        if fp16:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        return [
            (strategy.experimental_distribute_dataset(ds)
             if not isinstance(ds, tf.distribute.DistributedDataset) else ds)
            for ds in datasets]
    else:
        return datasets


def distribute_datasets(datasets):
    strategy = tf.distribute.get_strategy()
    return [
        (strategy.experimental_distribute_dataset(ds)
         if not isinstance(ds, tf.distribute.DistributedDataset) else ds)
        for ds in datasets]


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


def get_colab_runtime_version():
    # noinspection PyUnresolvedReferences
    from cloud_tpu_client import Client
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    client = Client(tpu="grpc://" + tpu_address)
    return client.runtime_version()


def set_colab_runtime_version(version=tf.__version__):
    # noinspection PyUnresolvedReferences
    from cloud_tpu_client import Client
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    client = Client(tpu="grpc://" + tpu_address)
    client.configure_tpu_version(version, restart_type='ifNeeded')
    client.wait_for_healthy()


def auth():
    # noinspection PyUnresolvedReferences
    from google.colab import auth
    auth.authenticate_user()


def strategy_run(strategy, fn, args):
    if strategy is not None:
        return strategy.run(fn, args=args)
    else:
        return fn(*args)


def is_tpu_strategy(strategy):
    if strategy is None:
        return False
    return "TPUStrategy" in type(strategy).__name__


def is_mirrored_strategy(strategy):
    if strategy is None:
        return False
    return "MirroredStrategy" in type(strategy).__name__


def is_distribute_strategy(strategy):
    return is_tpu_strategy(strategy) or is_mirrored_strategy(strategy)


def parse_strategy(strategy='auto') -> Optional[tf.distribute.Strategy]:
    if strategy is not None:
        if strategy == 'auto':
            strategy = tf.distribute.get_strategy()
        if not is_distribute_strategy(strategy):
            strategy = None
    return strategy


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


def set_gpu_thread_mode_and_count(num_gpus, gpu_thread_mode='gpu_private', per_gpu_thread_count=None):
    cpu_count = multiprocessing.cpu_count()

    per_gpu_thread_count = per_gpu_thread_count or 2
    os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)

    total_gpu_thread_count = per_gpu_thread_count * num_gpus
    num_runtime_threads = num_gpus
    datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads, num_gpus * 8)
    return datasets_num_private_threads
