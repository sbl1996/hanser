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
    'train': 8,
    'validation': 8
}

_SHUFFLE_BUFFER = 10000


def get_classes():
    return np.array(
        [700, 3, 590, 811, 889, 479, 433, 908, 2, 259, 295, 517, 1,
         209, 516, 790, 587, 758, 885, 613, 194, 332, 25, 218, 705, 939,
         191, 388, 579, 718, 229, 144, 305, 543, 875, 89, 658, 822, 205,
         76, 51, 660, 482, 624, 258, 341, 707, 448, 166, 352, 192, 399,
         898, 247, 589, 883, 909, 182, 732, 848, 737, 635, 460, 178, 221,
         782, 5, 450, 468, 656, 979, 773, 964, 703, 959, 987, 946, 88,
         925, 874, 48, 287, 969, 60, 952, 184, 560, 383, 717, 7, 973,
         407, 870, 414, 897, 457, 588, 56, 815, 813, 272, 706, 271, 591,
         616, 11, 98, 296, 755, 115, 975, 854, 625, 228, 290, 855, 237,
         395, 421, 446])


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
        os.path.join(data_dir, 'train-%05d-of-00008' % i)
        for i in range(NUM_FILES['train'])]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00008' % i)
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
