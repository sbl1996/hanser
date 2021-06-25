import math
from toolz import curry
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import tensorflow as tf
from hanser.datasets import prepare
from hanser.transform import to_tensor, normalize

@curry
def transform(x, y, training):
    x, y = to_tensor(x, y, vmax=16)
    x = normalize(x, [4.8842], [6.0168])
    return x, y

X, y = load_digits(n_class=10, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

batch_size = 128
eval_batch_size = batch_size * 2
n_train, n_test = len(X_train), len(X_test)
steps_per_epoch, test_steps = n_train // batch_size, math.ceil(n_test / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

ds_train = prepare(ds, batch_size, transform=transform(training=True), training=True, buffer_size=n_train)
ds_test = prepare(ds_test, eval_batch_size, transform=transform(training=False), training=False)

