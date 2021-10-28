import math
import os
import numpy as np
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


def get_dataset(filenames):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.interleave(
        tf.data.TFRecordDataset, cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
    ds = ds.map(parse_example)
    return ds


def random_classes(num_classes):
    cs = np.arange(1, 1 + 1000)
    np.random.shuffle(cs)
    return cs[:num_classes]


def collect(ds, classes, examples_per_class):
    class_samples = {}
    for c in classes:
        class_samples[c] = []

    count = 0
    for x, y in ds:
        y = y.numpy()
        if y in class_samples and len(class_samples[y]) < examples_per_class:
            class_samples[y].append(x.numpy())
            count += 1
            print(count)
            if count >= examples_per_class * len(classes):
                break
    return class_samples


def _process_dataset(records, output_directory, prefix, num_shards=8):
    chunksize = int(math.ceil(len(records) / num_shards))
    for shard in range(num_shards):
        chunk_records = records[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        writer = tf.io.TFRecordWriter(output_file)
        for r in chunk_records:
            writer.write(r)
        writer.close()
        print('Finished writing file: %s' % output_file)


c_train = 50
c_val = 20
num_classes = 120
examples_per_class = c_train + c_val

train_files = []
ds = get_dataset(train_files[:200])
classes = random_classes(num_classes)
class_samples = collect(ds, classes, examples_per_class)

output_directory = "tfrecords"
# !rm -rf $output_directory && mkdir $output_directory

records = [ r for rs in class_samples.values() for r in rs[:c_train] ]
_process_dataset(records, output_directory, "train")

records = [ r for rs in class_samples.values() for r in rs[-c_val:] ]
_process_dataset(records, output_directory, "validation")

# !gcloud auth activate-service-account --key-file "drive/MyDrive/private/xxxx.json"
# !gsutil cp -r $output_directory/* "gs://datasets/ImageNet_c120/"
