import math
import tensorflow as tf
import tensorflow_datasets as tfds
from hanser.datasets import prepare


def decode(example):
    image_id = example['image/filename']
    str_len = tf.strings.length(image_id)
    image_id = tf.strings.to_number(
        tf.strings.substr(image_id, str_len - 10, 6),
        out_type=tf.int32
    )
    image_id = tf.where(str_len == 10, image_id + 10000, image_id)

    image = tf.cast(example['image'], tf.float32)
    objects = {
        "gt_bbox": example['objects']['bbox'],
        'gt_label': tf.cast(example['objects']['label'] + 1, tf.int32),
        'is_difficult': example['objects']['is_difficult']
    }
    return image, objects, image_id


NUM_EXAMPLES = {
    'train': 16551,
    'val': 4952,
}

def make_voc_dataset(
    batch_size, eval_batch_size, transform, data_dir=None, drop_remainder=None):
    n_train, n_val = NUM_EXAMPLES['train'], NUM_EXAMPLES['val']
    steps_per_epoch = n_train // batch_size
    if drop_remainder:
        val_steps = n_val // eval_batch_size
    else:
        val_steps = math.ceil(n_val / eval_batch_size)

    ds_train1 = tfds.load("voc/2007", split=f"train+validation", data_dir=data_dir,
                          shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train2 = tfds.load("voc/2012", split=f"train+validation", data_dir=data_dir,
                          shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train = ds_train1.concatenate(ds_train2)
    ds_val = tfds.load("voc/2007", split=f"test", data_dir=data_dir,
                       shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=True)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps


def make_voc_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, transform, data_dir=None, prefetch=True):
    steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

    ds_train = tfds.load("voc/2012", split=f"train[:{n_train}]", data_dir=data_dir,
                         shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_val = tfds.load("voc/2012", split=f"train[:{n_val}]", data_dir=data_dir,
                       shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=True, prefetch=prefetch)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=False,
                     prefetch=prefetch)
    return ds_train, ds_val, steps_per_epoch, val_steps