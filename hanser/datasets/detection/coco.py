import math
import tensorflow_datasets as tfds
from hanser.datasets import prepare
from hanser.datasets.detection.general import decode

LABEL_MAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


def label_map(c):
    return LABEL_MAP[c]


NUM_EXAMPLES = {
    'train': 118287,
    'validation': 5000,
    'test': 40670,
}


def make_dataset(
    batch_size, eval_batch_size, transform, data_dir=None, drop_remainder=None):
    n_train, n_val = NUM_EXAMPLES['train'], NUM_EXAMPLES['validation']
    steps_per_epoch = n_train // batch_size
    if drop_remainder:
        val_steps = n_val // eval_batch_size
    else:
        val_steps = math.ceil(n_val / eval_batch_size)

    read_config = tfds.ReadConfig(try_autocache=False, skip_prefetch=True)
    ds_train = tfds.load("coco/2017", split=f"train", data_dir=data_dir,
                          shuffle_files=True, read_config=read_config)
    ds_val = tfds.load("coco/2017", split=f"validation", data_dir=data_dir,
                       shuffle_files=False, read_config=read_config)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=False)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps


def make_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, transform, data_dir=None, drop_remainder=None):
    steps_per_epoch = n_train // batch_size
    if drop_remainder:
        val_steps = n_val // eval_batch_size
    else:
        val_steps = math.ceil(n_val / eval_batch_size)

    read_config = tfds.ReadConfig(try_autocache=False, skip_prefetch=True)
    train_split = f"validation[:{n_train}]" if n_train != 5000 else 'validation'
    val_split = f"validation[:{n_val}]" if n_val != 5000 else 'validation'
    ds_train = tfds.load("coco/2017", split=train_split, data_dir=data_dir,
                          shuffle_files=False, read_config=read_config)
    ds_val = tfds.load("coco/2017", split=val_split, data_dir=data_dir,
                       shuffle_files=False, read_config=read_config)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=False)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps
