from hanser.datasets import prepare
from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.transform import pad, to_tensor, normalize

@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    # label = tf.one_hot(label, 10)

    return image, label

ds = tfds.load("mnist", split='train', try_gcs=False, download=True, shuffle_files=True, as_supervised=True)
ds_train = prepare(ds, 4, transform(training=True), training=False, buffer_size=8)

it = iter(ds_train)
next(it)[1]