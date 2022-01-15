import os
import math
import functools
import tensorflow as tf
from hanser.datasets import prepare
from hanser.datasets.classification.imagenet_classes import IMAGENET_CLASSES

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

NUM_FILES = {
    'train': 1024,
    'validation': 128
}

_SHUFFLE_BUFFER = 1251 * 1024


def parse_example_proto(example_serialized):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                   default_value=-1),
    }

    features = tf.io.parse_single_example(serialized=example_serialized,
                                          features=feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label


def parse_and_transform(transform, training):
    def fn(x):
        image, label = parse_example_proto(x)
        return transform(image, label, training)
    return fn


def make_train_split(
    batch_size, transform, filenames, n_batches_per_step=1, buffer_size=None, **kwargs):
    training = True
    if buffer_size is None:
        buffer_size = _SHUFFLE_BUFFER

    batch_size = batch_size // n_batches_per_step

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(buffer_size=len(filenames))
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    dataset = dataset.map(parse_example_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    transform = functools.partial(transform, training=training)

    ds = prepare(dataset, batch_size, transform, training=training, buffer_size=buffer_size,
                 cache=True, prefetch=True, repeat=True, drop_remainder=True, **kwargs)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    n = NUM_IMAGES['train']
    chunksize = math.ceil(n / NUM_FILES['train'])
    n = min(len(filenames) * chunksize, n)
    if 'repeat' in kwargs and type(kwargs['repeat']) == int:
        n *= kwargs['repeat']
    if 'aug_repeats' in kwargs and type(kwargs['aug_repeats']) == int:
        n *= kwargs['aug_repeats']
    steps = n // (batch_size * n_batches_per_step)
    return ds, steps


def make_eval_split(batch_size, transform, load_path, drop_remainder=True, **kwargs):

    dataset = tf.data.experimental.load(load_path)
    dataset = dataset.map(parse_example_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    transform = functools.partial(transform, training=False)

    ds = prepare(dataset, batch_size, transform, training=False, cache=True,
                 prefetch=True, repeat=True, drop_remainder=drop_remainder, **kwargs)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    ds = ds.with_options(options)

    n = NUM_IMAGES['validation']
    if drop_remainder:
        steps = n // batch_size
    else:
        steps = math.ceil(n / batch_size)
    return ds, steps


def make_imagenet_dataset(
    batch_size, eval_batch_size, transform, train_files=None, eval_files=None,
    zip_transform=None, batch_transform=None, aug_repeats=None, drop_remainder=None,
    n_batches_per_step=1, **kwargs):

    assert isinstance(eval_files, str)

    ds_train, steps_per_epoch = make_train_split(
        batch_size, transform, train_files, n_batches_per_step,
        zip_transform=zip_transform, batch_transform=batch_transform,
        aug_repeats=aug_repeats, **kwargs)
    ds_eval, eval_steps = make_eval_split(
        eval_batch_size, transform, eval_files, drop_remainder=drop_remainder, **kwargs)
    return ds_train, ds_eval, steps_per_epoch, eval_steps
