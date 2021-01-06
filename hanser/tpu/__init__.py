import os
import multiprocessing

import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica
import tensorflow.keras.mixed_precision as mixed_precision


def setup(datasets, fp16=True, device='auto', cross_device_ops=None):
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
            mixed_precision.set_global_policy(policy)
        return datasets
    elif isinstance(device, list) or device == 'GPUs':
        tf.distribute.experimental_set_strategy(strategy)
        if fp16:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
        return [
            (strategy.experimental_distribute_dataset(ds)
             if not isinstance(ds, tf.distribute.DistributedDataset) else ds)
            for ds in datasets]
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
    # noinspection PyUnresolvedReferences
    from google.colab import auth
    auth.authenticate_user()


def local_results(strategy, values):
    if isinstance(values, PerReplica):
        return strategy.experimental_local_results(values)
    elif isinstance(values, (list, tuple)):
        return values.__class__(local_results(strategy, v) for v in values)
    elif isinstance(values, dict):
        return {k: local_results(strategy, v) for k, v in values.items()}
    else:
        return values


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
