import math
from typing import Type, Callable, Optional

from hanser.datasets import prepare
from hanser.datasets.classification.general import ImageListBuilder
from hanser.datasets.tfds.helper import load as tfds_load


def make_fg_dataset(
    dataset_cls: Type[ImageListBuilder],
    batch_size: int,
    eval_batch_size: int,
    transform: Callable,
    data_dir: Optional[str] = None,
    train_split: str = 'train',
    eval_split: str = 'val',
    zip_transform=None, batch_transform=None):
    dataset_name = dataset_cls.name
    n_train, n_val = dataset_cls.SPLITS[train_split], dataset_cls.SPLITS[eval_split]
    steps_per_epoch = n_train // batch_size
    eval_steps = math.ceil(n_val / eval_batch_size)

    ds_train = tfds_load(dataset_name, split=train_split, shuffle_files=True,
                         data_dir=data_dir)
    ds_eval = tfds_load(dataset_name, split=eval_split, shuffle_files=False,
                        data_dir=data_dir)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       batch_transform=batch_transform, zip_transform=zip_transform,
                       training=True, repeat=True)
    ds_eval = prepare(ds_eval, eval_batch_size, transform(training=False),
                      training=False, repeat=False)
    return ds_train, ds_eval, steps_per_epoch, eval_steps


def make_fg_dataset_sub(
    dataset_cls: Type[ImageListBuilder],
    n_train: int,
    n_val: int,
    batch_size: int,
    eval_batch_size: int,
    transform: Callable,
    data_dir: Optional[str] = None,
    zip_transform=None, batch_transform=None):
    dataset_name = dataset_cls.name
    steps_per_epoch = n_train // batch_size
    eval_steps = math.ceil(n_val / eval_batch_size)

    ds_train = tfds_load(dataset_name, split=f"train[:{n_train}]", shuffle_files=True,
                         data_dir=data_dir)
    ds_eval = tfds_load(dataset_name, split=f"train[:{n_val}]", shuffle_files=False,
                        data_dir=data_dir)
    ds_train = prepare(ds_train, batch_size, transform(training=True),
                       batch_transform=batch_transform, zip_transform=zip_transform,
                       training=True, repeat=True)
    ds_eval = prepare(ds_eval, eval_batch_size, transform(training=False),
                      training=False, repeat=False)
    return ds_train, ds_eval, steps_per_epoch, eval_steps