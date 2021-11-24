import os
import multiprocessing

import tensorflow as tf
import tensorflow.keras.mixed_precision as mixed_precision


def setup_gpu(fp16=True):
    assert has_gpu()
    tf.keras.backend.clear_session()
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    if fp16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)


def has_gpu():
    return len(tf.config.list_physical_devices('GPU')) > 0


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
