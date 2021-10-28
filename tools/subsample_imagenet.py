import tensorflow as tf

feature_map = {
    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
    'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                default_value=-1),
}


def parse_example(example):
    features = tf.io.parse_single_example(serialized=example,
                                          features=feature_map)
    return example, features['image/class/label']


ds = tf.data.Dataset.from_tensor_slices([])
ds = ds.interleave(
    tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
ds = ds.map(parse_example)

filename = 'test.tfrecord'
writer = tf.io.TFRecordWriter(filename)
c = 0
for x, y in ds:
    if y.numpy() == 0:
        writer.write(x.numpy())
        c += 1