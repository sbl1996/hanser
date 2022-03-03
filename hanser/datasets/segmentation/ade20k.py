import math
import tensorflow_datasets as tfds
from hanser.datasets import prepare
import hanser.datasets.tfds.segmentation.ade20k

NUM_EXAMPLES = {
    'train': 20210,
    'validation': 2000,
}


def make_dataset(
    batch_size, eval_batch_size, transform, data_dir=None,
    drop_remainder=None, repeat=True):
    n_train, n_val = NUM_EXAMPLES['train'], NUM_EXAMPLES['validation']
    steps_per_epoch = n_train // batch_size
    if drop_remainder:
        val_steps = n_val // eval_batch_size
    else:
        val_steps = math.ceil(n_val / eval_batch_size)

    read_config = tfds.ReadConfig(try_autocache=False, skip_prefetch=True)
    ds_train = tfds.load("ade_20_k", split=f"train", data_dir=data_dir,
                          shuffle_files=True, read_config=read_config)
    ds_val = tfds.load("ade_20_k", split=f"validation", data_dir=data_dir,
                       shuffle_files=False, read_config=read_config)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       training=True, repeat=True)
    ds_val = prepare(ds_val, eval_batch_size, transform(training=False),
                     training=False, repeat=repeat, drop_remainder=drop_remainder)
    return ds_train, ds_val, steps_per_epoch, val_steps