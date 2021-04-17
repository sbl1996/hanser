import math
import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.datasets import prepare
try:
    import hanser.datasets.tfds.detection.svhn
except ValueError:
    pass


def decode(example):
    image_id = example['image/id']
    image = tf.cast(example['image'], tf.float32)
    objects = {
        "bbox": example['objects']['bbox'],
        'label': tf.cast(example['objects']['label'] + 1, tf.int32),
    }
    return image, objects, image_id


NUM_EXAMPLES = {
    'train': 33402,
    'test': 13068,
}

def make_svhn_dataset(
    batch_size, eval_batch_size, transform, data_dir=None, drop_remainder=None):
    n_train, n_val = NUM_EXAMPLES['train'], NUM_EXAMPLES['test']
    steps_per_epoch = n_train // batch_size
    if drop_remainder:
        val_steps = n_val // eval_batch_size
    else:
        val_steps = math.ceil(n_val / eval_batch_size)

    ds_train = tfds.load("svhn", split=f"train", data_dir=data_dir,
                         shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_val = tfds.load("svhn", split=f"test", data_dir=data_dir,
                       shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=False)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps


def make_svhn_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, transform, data_dir=None, prefetch=True):
    steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

    ds_train = tfds.load("svhn", split=f"train[:{n_train}]", data_dir=data_dir,
                         shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_val = tfds.load("svhn", split=f"train[:{n_val}]", data_dir=data_dir,
                       shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=False, prefetch=prefetch)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=False,
                     prefetch=prefetch)
    return ds_train, ds_val, steps_per_epoch, val_steps