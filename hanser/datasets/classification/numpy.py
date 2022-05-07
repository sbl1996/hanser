import math
import numpy as np

import tensorflow as tf

from hanser.datasets.utils import prepare


def subsample(*arrays, ratio):
    lens = [len(a) for a in arrays]
    assert len(set(lens)) == 1
    n = int(lens[0] * ratio)
    indices = np.random.choice(n, n, replace=False)
    sub_arrays = tuple(a[indices] for a in arrays)
    return sub_arrays


def make_numpy_dataset(
    x_train, y_train, x_test, y_test,
    batch_size, eval_batch_size, transform,
    sub_ratio=None, aug_repeats=None, drop_remainder=None, **kwargs):
    if sub_ratio is not None:
        x_train, y_train = subsample(x_train, y_train, ratio=sub_ratio)
        x_test, y_test = subsample(x_test, y_test, ratio=sub_ratio)
    n_train, n_test = len(x_train), len(x_test)
    if aug_repeats is not None:
        n_train *= aug_repeats
    steps_per_epoch = n_train // batch_size
    test_steps = math.ceil(n_test / eval_batch_size)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_train = prepare(ds_train, batch_size, transform(training=True), training=True, buffer_size=n_train,
                       aug_repeats=aug_repeats, **kwargs)
    ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False,
                      drop_remainder=drop_remainder)
    return ds_train, ds_test, steps_per_epoch, test_steps
