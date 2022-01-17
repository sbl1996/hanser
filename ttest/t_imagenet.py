import os
from functools import partial
import tensorflow as tf

# train_files = [f"{os.getenv('GCS_BUCKET')}/ImageNet/train-%05d-of-01024" % i for i in range(1024)]
# eval_files = [f"{os.getenv('GCS_BUCKET')}/ImageNet/validation-%05d-of-00128" % i for i in range(128)]
# filenames = eval_files
# dataset = tf.data.Dataset.from_tensor_slices(filenames)
# dataset = dataset.shuffle(buffer_size=len(filenames))
# dataset = dataset.interleave(
#     partial(tf.data.TFRecordDataset, buffer_size=8<<20),
#     block_length=16,
#     cycle_length=16,
#     num_parallel_calls=tf.data.experimental.AUTOTUNE,
#     deterministic=False)
#
# num_shards = 128
# def shard_func(dataset):
#     return tf.random.uniform((), 0, num_shards, dtype=tf.int64)
#


#
# def parse_example_proto(example_serialized):
#     feature_map = {
#         'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
#                                                default_value=''),
#         'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
#                                                    default_value=-1),
#     }
#
#     features = tf.io.parse_single_example(serialized=example_serialized,
#                                           features=feature_map)
#     label = tf.cast(features['image/class/label'], dtype=tf.int32)
#
#     return features['image/encoded'], label
#

eval_files = f"{os.getenv('GCS_BUCKET')}/ImageNet/validation_saved"
ds = tf.data.experimental.load(eval_files, element_spec=tf.TensorSpec((), dtype=tf.string))


# ds = ds.map(parse_example_proto, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.cache()

tf.data.experimental.save(ds, "./valid")

it = iter(ds_eval)

s = next(it)

for i in range(50000-1):
# for i in range(1281167-1):
    s = next(it)
    # s = s.numpy()
    s = s[0].numpy()
    # s = s.numpy()
    if i % 10000 == 0:
        print(i)