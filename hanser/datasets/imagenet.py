import os
import math
import functools
import tensorflow as tf
from hanser.datasets import prepare
from hanser.datasets.classification.imagenet_classes import IMAGENET_CLASSES
from hhutil.io import fmt_path

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

NUM_FILES = {
    'train': 1024,
    'validation': 128
}

_SHUFFLE_BUFFER = 10000


def _get_filenames(data_dir, training):
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


def parse_and_transform(transform, training):
    def fn(x):
        image, label = parse_example_proto(x)
        return transform(image, label, training)
    return fn


def make_imagenet_dataset_split(
    batch_size, transform, filenames, split, training=None,
    cache_parsed=False, drop_remainder=None, repeat=True,
    n_batches_per_step=1, buffer_size=None, cache=True, **kwargs):
    assert split in NUM_IMAGES.keys()

    if training is None:
        training = split == 'train'

    if drop_remainder is None:
        drop_remainder = training

    if n_batches_per_step != 1:
        assert training

    if buffer_size is None:
        buffer_size = _SHUFFLE_BUFFER

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

    ds = prepare(dataset, batch_size, transform, training=training, buffer_size=buffer_size,
                 cache=cache, prefetch=True, repeat=repeat, drop_remainder=drop_remainder, **kwargs)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.max_intra_op_parallelism = 1
    # options.experimental_slack = True
    # options.experimental_threading.private_threadpool_size = 48
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    ds = ds.with_options(options)

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
    n_batches_per_step=1, cache_eval=True, **kwargs):

    if train_files is None:
        train_files = _get_filenames(data_dir, training=True)
    if eval_files is None:
        eval_files = _get_filenames(data_dir, training=False)

    ds_train, steps_per_epoch = make_imagenet_dataset_split(
        batch_size, transform, train_files, 'train', training=True,
        zip_transform=zip_transform, batch_transform=batch_transform,
        aug_repeats=aug_repeats, n_batches_per_step=n_batches_per_step, **kwargs)
    ds_eval, eval_steps = make_imagenet_dataset_split(
        eval_batch_size, transform, eval_files, 'validation', training=False,
        drop_remainder=drop_remainder, n_batches_per_step=1, cache=cache_eval, **kwargs)
    return ds_train, ds_eval, steps_per_epoch, eval_steps


def get_files(remote_dir, local_dir, filenames):
    assert not (remote_dir is None and local_dir is None)
    if local_dir is None:
        return [os.path.join(remote_dir, filename) for filename in filenames]
    if remote_dir is None:
        return [os.path.join(local_dir, filename) for filename in filenames]
    local_dir = fmt_path(local_dir)
    fps = []
    for filename in filenames:
        local_path = local_dir / filename
        if local_path.exists():
            fps.append(str(local_path))
        else:
            remote_path = os.path.join(remote_dir, filename)
            fps.append(remote_path)
    return fps