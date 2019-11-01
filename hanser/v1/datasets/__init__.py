import tensorflow as tf


def prepare(ds, preprocess, batch_size, training=True, buffer_size=None, drop_remainder=None):
    if drop_remainder is None:
        drop_remainder = training
    ds = ds.cache()
    if training:
        ds = ds.shuffle(buffer_size or 1000)
        ds = ds.repeat()
    ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    else:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder).repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds