import math
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import get_file

from hanser.datasets.utils import prepare

def load_mnist(path='mnist.npz'):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file(
        path,
        origin=origin_folder + 'mnist.npz',
        file_hash=
        '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

def make_mnist_dataset(batch_size, eval_batch_size, transform, zip_transform=None, sub_ratio=None):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, x_test = x_train[:, :, :, None], x_test[:, :, :, None]

    n_train, n_test = len(x_train), len(x_test)
    if sub_ratio is not None:
        n_train = int(n_train * sub_ratio)
        indices = np.random.choice(n_train, n_train, replace=False)
        x_train, y_train = x_train[indices], y_train[indices]

        n_test = int(n_test * sub_ratio)
        indices = np.random.choice(n_test, n_test, replace=False)
        x_test, y_test = x_test[indices], y_test[indices]
    steps_per_epoch = n_train // batch_size
    test_steps = math.ceil(n_test / eval_batch_size)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=n_train,
                       zip_transform=zip_transform)
    ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)
    return ds_train, ds_test, steps_per_epoch, test_steps