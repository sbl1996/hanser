from toolz import curry

import tensorflow as tf


@curry
def batch_to_zip_transform(image, label, zip_transform):
    return zip_transform((image[0], label[0]), (image[1], label[1]))


def prepare(ds, batch_size, transform=None, training=True, buffer_size=1024, drop_remainder=None, cache=True,
            zip_transform=None, batch_transform=None):
    if drop_remainder is None:
        drop_remainder = training
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(buffer_size)
        ds = ds.repeat()
    if transform:
        ds = ds.map(transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        if zip_transform:
            # ds = ds.batch(2, drop_remainder=True).map(
            #     batch_to_zip_transform(zip_transform=zip_transform), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            ds = tf.data.Dataset.zip((ds, ds)).map(zip_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if batch_transform:
            ds = ds.map(batch_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder).repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds