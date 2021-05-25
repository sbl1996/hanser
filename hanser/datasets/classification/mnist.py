import numpy as np
from tensorflow.keras.utils import get_file

from hanser.datasets.classification.numpy import make_numpy_dataset


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
        x_train, x_test = x_train[:, :, :, None], x_test[:, :, :, None]
        return (x_train, y_train), (x_test, y_test)


def make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=None, **kwargs):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    return make_numpy_dataset(
        x_train, y_train, x_test, y_test,
        batch_size, eval_batch_size, transform, sub_ratio, **kwargs)