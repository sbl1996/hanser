import tensorflow as tf


def prepare(ds, batch_size, transform=None, training=True, buffer_size=1024,
            drop_remainder=None, cache=True, repeat=True,
            zip_transform=None, batch_transform=None):
    if drop_remainder is None:
        drop_remainder = training
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(buffer_size)
        if repeat:
            ds = ds.repeat()
    if transform:
        ds = ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        if zip_transform:
            ds = tf.data.Dataset.zip((ds, ds)).map(zip_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if batch_transform:
            ds = ds.map(batch_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if batch_transform:
            ds = ds.map(batch_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if repeat:
            ds = ds.repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds