import os
import math
import functools
import numpy as np
import tensorflow as tf
from hanser.datasets import prepare
from hanser.datasets.classification.imagenet_classes import IMAGENET_CLASSES
from hanser.datasets.imagenet import parse_example_proto

NUM_IMAGES = {
    'train': 6000,
    'validation': 2400,
}

NUM_FILES = {
    'train': 10,
    'validation': 4
}

_SHUFFLE_BUFFER = 10000


def get_classes():
    return np.array(
        [620, 406, 973, 694, 871, 246, 261, 125, 36, 312, 58,
         644, 66, 101, 141, 744, 597, 165, 127, 472, 486, 659,
         727, 523, 119, 983, 310, 608, 823, 987, 1000, 366, 479,
         812, 345, 94, 660, 468, 524, 79, 402, 839, 542, 442,
         588, 138, 634, 504, 907, 178, 675, 943, 525, 151, 962,
         697, 565, 42, 349, 328, 493, 970, 267, 41, 158, 496,
         25, 906, 478, 671, 148, 801, 913, 776, 950, 354, 359,
         552, 448, 180, 520, 459, 770, 955, 289, 648, 609, 421,
         53, 265, 683, 902, 103, 306, 452, 957, 711, 314, 742,
         384, 695, 686, 28, 775, 625, 613, 255, 264, 899, 187,
         10, 614, 17, 984, 567, 161, 752, 971, 633, 510])


def get_label_map(classes):
    m = np.zeros(1001, dtype=np.int)
    for i, c in enumerate(classes):
        m[c] = i + 1
    return m


def map_label(label):
    label_map = get_label_map(get_classes())
    return tf.gather(tf.constant(label_map, label.dtype), label)


def get_filenames(data_dir, training):
  if training:
    return [
        os.path.join(data_dir, 'train-%05d-of-00010' % i)
        for i in range(NUM_FILES['train'])]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00004' % i)
        for i in range(NUM_FILES['validation'])]


def parse_and_transform(transform, training):
    def fn(x):
        image, label = parse_example_proto(x)
        label = map_label(label)
        return transform(image, label, training)
    return fn


def make_imagenet_dataset_split(
    batch_size, transform, filenames, split, training=None,
    cache_parsed=False, drop_remainder=None, repeat=None,
    n_batches_per_step=1, **kwargs):
    assert split in NUM_IMAGES.keys()

    if training is None:
        training = split == 'train'

    if drop_remainder is None:
        drop_remainder = training

    if repeat is None:
        repeat = training

    if n_batches_per_step != 1:
        assert training

    batch_size = batch_size // n_batches_per_step

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if training:
        dataset = dataset.shuffle(buffer_size=len(filenames))
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    if cache_parsed:
        dataset = dataset.map(parse_example_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        transform = functools.partial(transform, training=training)
    else:
        transform = parse_and_transform(transform, training)
    ds = prepare(dataset, batch_size, transform, training=training, buffer_size=_SHUFFLE_BUFFER,
                 cache=True, prefetch=True, repeat=repeat, drop_remainder=drop_remainder, **kwargs)

    n = NUM_IMAGES[split]
    chunksize = math.ceil(n / NUM_FILES[split])
    n = min(len(filenames) * chunksize, n)
    if 'repeat' in kwargs and type(kwargs['repeat']) == int:
        n *= kwargs['repeat']
    if 'aug_repeats' in kwargs and type(kwargs['aug_repeats']) == int:
        n *= kwargs['aug_repeats']
    if drop_remainder:
        steps = n // (batch_size * n_batches_per_step)
    else:
        steps = math.ceil(n / batch_size)

    return ds, steps


def make_imagenet_dataset(
    batch_size, eval_batch_size, transform, data_dir=None, train_files=None, eval_files=None,
    zip_transform=None, batch_transform=None, aug_repeats=None, drop_remainder=None,
    n_batches_per_step=1, **kwargs):

    if train_files is None:
        train_files = get_filenames(data_dir, training=True)
    if eval_files is None:
        eval_files = get_filenames(data_dir, training=False)

    ds_train, steps_per_epoch = make_imagenet_dataset_split(
        batch_size, transform, train_files, 'train', training=True,
        zip_transform=zip_transform, batch_transform=batch_transform,
        aug_repeats=aug_repeats, n_batches_per_step=n_batches_per_step, **kwargs)
    ds_eval, eval_steps = make_imagenet_dataset_split(
        eval_batch_size, transform, eval_files, 'validation', training=False,
        drop_remainder=drop_remainder, n_batches_per_step=1, **kwargs)
    return ds_train, ds_eval, steps_per_epoch, eval_steps
