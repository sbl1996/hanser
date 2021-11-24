import tensorflow as tf


def make_repeat_fn(n):
    def fn(*args):
        return tf.data.Dataset.from_tensor_slices(tuple([arg for _ in range(n)] for arg in args))
    return fn


def prepare(ds: tf.data.Dataset, batch_size, transform=None, training=True, buffer_size=1024,
            drop_remainder=None, cache=True, repeat=True, prefetch=True,
            zip_transform=None, batch_transform=None, aug_repeats=None):

    if drop_remainder is None:
        drop_remainder = training
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(buffer_size)
        if aug_repeats is not None:
            # assert len(ds.element_spec) == 2
            ds = ds.flat_map(make_repeat_fn(aug_repeats))
        if type(repeat) == int:
            ds = ds.repeat(repeat)
        elif repeat:
            ds = ds.repeat()
    if transform:
        ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        if zip_transform:
            ds = tf.data.Dataset.zip((ds, ds)).map(zip_transform, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if batch_transform:
            ds = ds.map(batch_transform, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        if batch_transform:
            ds = ds.map(batch_transform, num_parallel_calls=tf.data.AUTOTUNE)
        if repeat:
            ds = ds.repeat()
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds