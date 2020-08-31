import tensorflow as tf
from hanser.datasets import prepare

ds = tf.data.Dataset.range(10)
ds_p = prepare(ds, training=False, batch_size=4, drop_remainder=False)
it = iter(ds_p)