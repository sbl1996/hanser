import io
import math

from PIL import Image

import numpy as np
from hhutil.io import fmt_path
import tensorflow as tf


def read_image(fp):
    fp = fmt_path(fp)
    b = fp.read_bytes()
    img = tf.image.decode_image(b)
    height, width = img.shape[:2]
    if len(img.shape) == 3:
        channels = img.shape[2]
    else:
        channels = 1
    return b, (width, height, channels)


def read_label(fp, format='PNG'):
    arr = np.array(Image.open(fp))
    seg = Image.fromarray(arr)
    size = seg.size
    data = io.BytesIO()
    seg.save(data, format=format)
    data = data.getvalue()
    return data, size


def _bytes_feature(value):
    if isinstance(value, str):
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_label_to_tfexample(imgae_data, filename, image_format, height, width, channels, label_data, label_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(imgae_data),
        'image/filename': _bytes_feature(filename),
        'image/format': _bytes_feature(image_format),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/segmentation/class/encoded': _bytes_feature(label_data),
        'image/segmentation/class/format': _bytes_feature(label_format),
    }))


def parse_tfexample_to_img_seg(example_proto):
    features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
    }
    return tf.io.parse_single_example(example_proto, features)


def convert_segmentation_dataset(
    split_f, output_dir, image_dir, label_dir,
    image_format='jpg', label_format='png', num_shards=4,
    image_stem_transform=lambda x: x, label_stem_transform=lambda x: x):
    split_f = fmt_path(split_f)
    output_dir = fmt_path(output_dir)
    image_dir = fmt_path(image_dir)
    label_dir = fmt_path(label_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    split = split_f.stem
    filenames = split_f.read_text().splitlines()
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(num_shards)))

    print('Processing ' + split)

    for shard_id in range(num_shards):
        output_f = output_dir / ('%s-%05d-of-%05d.tfrecord' % (split, shard_id, num_shards))
        with tf.io.TFRecordWriter(str(output_f)) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                print('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id), end='')

                image_fp = image_dir / (image_stem_transform(filenames[i]) + '.' + image_format)
                image_data, (width, height, channels) = read_image(image_fp)

                label_fp = label_dir / (label_stem_transform(filenames[i]) + '.' + label_format)
                label_data, (label_width, label_height) = read_label(label_fp, format=label_format)

                if height != label_height or width != label_width:
                    raise RuntimeError('Shape mismatched between image and label.')

                example = image_label_to_tfexample(
                    image_data, filenames[i], image_format, height, width, channels, label_data, label_format)
                writer.write(example.SerializeToString())
        print()


def convert_numpy_dataset(X, y, split, output_dir, num_shards=4):
    assert len(X) == len(y)

    output_dir = fmt_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_examples = len(X)
    num_per_shard = int(math.ceil(num_examples / float(num_shards)))

    for shard_id in range(num_shards):
        output_f = output_dir / ('%s-%05d-of-%05d.tfrecord' % (split, shard_id, num_shards))
        with tf.io.TFRecordWriter(str(output_f)) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_examples)
            for i in range(start_idx, end_idx):
                print('\r>> Converting image %d/%d shard %d' % (i + 1, num_examples, shard_id), end='')
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(X[i].tobytes()),
                    'label': _int64_feature(y[i]),
                }))
                writer.write(example.SerializeToString())
        print()


def parse_numpy_example(example_proto):
    features = {
        'image': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.io.FixedLenFeature((), tf.int64, default_value=0),
    }
    return tf.io.parse_single_example(example_proto, features)