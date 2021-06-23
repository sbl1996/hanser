import math
import tensorflow_datasets as tfds
from hanser.datasets import prepare
import hanser.datasets.tfds.detection.cocoval

from hanser.datasets.detection.coco import label_map, decode

NUM_EXAMPLES = {
    'validation': 5000,
}


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
    ds_train = tfds.load("coco_val/2017", split=train_split, data_dir=data_dir,
                          shuffle_files=True, read_config=read_config)
    ds_val = tfds.load("coco_val/2017", split=val_split, data_dir=data_dir,
                       shuffle_files=False, read_config=read_config)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=True)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=False, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps
