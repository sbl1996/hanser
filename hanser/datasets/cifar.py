import os
import pickle

import numpy as np

import tensorflow_datasets as tfds
from tensorflow.python.keras.utils.data_utils import get_file


def load_cifar10_tfds():

    def load_data(split):
        ds = tfds.as_numpy(tfds.load('cifar10', split=split))
        x = []
        y = []
        for i, d in enumerate(ds):
            print(i)
            x.append(d['image'])
            y.append(d['label'])
        x = np.stack(x)
        y = np.stack(y)
        return x, y

    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')
    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath, label_key='labels'):

    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_cifar10():

    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        untar=True,
        file_hash=
        '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    y_train = np.array(y_train, dtype=np.int)
    y_test = np.array(y_test, dtype=np.int)

    return (x_train, y_train), (x_test, y_test)


def load_cifar100(label_mode='fine'):

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        untar=True,
        file_hash=
        '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7')
    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    y_train = np.array(y_train, dtype=np.int)
    y_test = np.array(y_test, dtype=np.int)

    return (x_train, y_train), (x_test, y_test)