import os
import math
import functools
import tensorflow as tf
from hanser.datasets import prepare
from hanser.datasets.imagenet_classes import IMAGENET_CLASSES

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

NUM_FILES = {
    'train': 1024,
    'validation': 128
}

_SHUFFLE_BUFFER = 10000


def get_filenames(data_dir, training):
  if training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(NUM_FILES['train'])]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(NUM_FILES['validation'])]


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


# def parse_and_transform(transform):
#     def fn(x, training):
#         image, label = parse_example_proto(x)
#         return transform(image, label, training)
#     return fn

# def parse_example_proto_and_decode(example_serialized):
#     image_buffer, label = parse_example_proto(example_serialized)
#     image_buffer = tf.reshape(image_buffer, shape=[])
#     image_buffer = tf.io.decode_jpeg(image_buffer, 3)
#     return image_buffer, label

# def input_fn(filenames, training, transform, batch_size,
#              batch_transform=None, zip_transform=None,
#              cache_dataset=True, cache_decoded_image=False,
#              repeat=False, buffer_size=_SHUFFLE_BUFFER):
#
#     dataset = tf.data.Dataset.from_tensor_slices(filenames)
#
#     if training:
#         dataset = dataset.shuffle(buffer_size=len(filenames))
#
#     dataset = dataset.interleave(
#         tf.data.TFRecordDataset,
#         cycle_length=10,
#         num_parallel_calls=tf.data.experimental.AUTOTUNE,
#         deterministic=False)
#
#     if training and cache_decoded_image:
#         dataset = dataset.map(
#             parse_example_proto_and_decode,
#             num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     if cache_dataset:
#         dataset = dataset.cache()
#
#     if training:
#         dataset = dataset.shuffle(buffer_size=buffer_size)
#     if type(repeat) == int:
#         dataset = dataset.repeat(repeat)
#     elif repeat:
#         dataset = dataset.repeat()
#
#     if training and cache_decoded_image:
#         parse_record_fn = transform
#     else:
#         parse_record_fn = parse_and_transform(transform)
#
#     map_fn = functools.partial(
#         parse_record_fn, training=training)
#
#     dataset = dataset.map(
#         map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     if training and zip_transform:
#         dataset = tf.data.Dataset.zip((dataset, dataset)).map(
#             zip_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.batch(batch_size, drop_remainder=training)
#     if training and batch_transform:
#         dataset = dataset.map(
#             batch_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#     dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#     return dataset


def parse_and_transform(transform, training):
    def fn(x):
        image, label = parse_example_proto(x)
        return transform(image, label, training)
    return fn


def make_imagenet_dataset_split(
    batch_size, transform, filenames, split, training=None,
    cache_decoded_image=False, cache_parsed=False, **kwargs):
    assert split in NUM_IMAGES.keys()

    if training is None:
        training = split == 'train'

    drop_remainder = training

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
                 cache=True, prefetch=True, repeat=False, **kwargs)

    n = NUM_IMAGES[split]
    chunksize = math.ceil(n / NUM_FILES[split])
    n = min(len(filenames) * chunksize, n)
    if 'repeat' in kwargs and type(kwargs['repeat']) == int:
        n *= kwargs['repeat']
    if 'aug_repeats' in kwargs and type(kwargs['aug_repeats']) == int:
        n *= kwargs['aug_repeats']
    if drop_remainder:
        steps = n // batch_size
    else:
        steps = math.ceil(n / batch_size)

    return ds, steps


def make_imagenet_dataset(
    batch_size, eval_batch_size, transform, data_dir=None, train_files=None, eval_files=None,
    zip_transform=None, batch_transform=None, aug_repeats=None, **kwargs):

    if train_files is None:
        train_files = get_filenames(data_dir, training=True)
    if eval_files is None:
        eval_files = get_filenames(data_dir, training=False)

    ds_train, steps_per_epoch = make_imagenet_dataset_split(
        batch_size, transform, train_files, 'train', training=True,
        zip_transform=zip_transform, batch_transform=batch_transform,
        aug_repeats=aug_repeats, **kwargs)
    ds_eval, eval_steps = make_imagenet_dataset_split(
        eval_batch_size, transform, eval_files, 'validation', training=False, **kwargs)
    return ds_train, ds_eval, steps_per_epoch, eval_steps
