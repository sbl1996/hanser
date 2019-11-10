import io
import math
from pathlib import Path

from PIL import Image

import numpy as np
import tensorflow as tf


def read_img(fp):
    b = fp.read_bytes()
    img = tf.image.decode_image(b)
    height, width = img.shape[:2]
    return b, (width, height)


def read_seg(fp, format='PNG'):
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
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def img_seg_to_tfexample(img_data, filename, image_format, height, width, seg_data, label_format):
    """Converts one image/segmentation pair to tf example.

    Args:
      img_data: string of image data.
      filename: image filename.
      height: image height.
      width: image width.
      seg_data: string of semantic segmentation data.

    Returns:
      tf example of one image/segmentation pair.
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(img_data),
        'image/filename': _bytes_feature(filename),
        'image/format': _bytes_feature(image_format),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(3),
        'image/segmentation/class/encoded': _bytes_feature(seg_data),
        'image/segmentation/class/format': _bytes_feature(label_format),
    }))


def parse_voc_example(example_proto):
    features = {
        'image':
            tf.io.FixedLenFeature((), tf.string),
        'image/filename':
            tf.io.FixedLenFeature((), tf.string),
        'objects/bbox':
            tf.io.FixedLenSequenceFeature((4,), tf.float32, allow_missing=True),
        'objects/is_difficult':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'objects/is_truncated':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'objects/label':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'objects/pose':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'labels':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
        'labels_no_difficult':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
    }
    data = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_image(data['image'])
    img.set_shape([None, None, 3])
    data['image'] = img
    return data


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


def _check_path(fp):
    return Path(fp).expanduser().absolute()


def convert_segmentation_dataset(split_f, output_dir, img_dir, seg_dir, image_format='jpg', label_format='png',
                                 num_shards=4):
    """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
    split_f = _check_path(split_f)
    output_dir = _check_path(output_dir)
    img_dir = _check_path(img_dir)
    seg_dir = _check_path(seg_dir)

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

                img_f = img_dir / (filenames[i] + '.' + image_format)
                img_data, (width, height) = read_img(img_f)

                seg_f = seg_dir / (filenames[i] + '.' + label_format)
                seg_data, (seg_width, seg_height) = read_seg(seg_f, format=label_format)

                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')

                example = img_seg_to_tfexample(
                    img_data, filenames[i], image_format, height, width, seg_data, label_format)
                writer.write(example.SerializeToString())
        print()


def convert_numpy_dataset(X, y, split, output_dir, num_shards=4):
    """Converts the specified dataset split to TFRecord format.

    Args:
        split: The dataset split (e.g., train, test).

    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    assert len(X) == len(y)

    output_dir = _check_path(output_dir)
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