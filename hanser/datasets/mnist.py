import os
import pickle

import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file


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