import tensorflow as tf

from hanser.datasets.tfrecord import _bytes_feature, _int64_feature, _float_list_feature


def to_tfexample(image, filename, image_format, height, width, label_data, label_format):
    """Converts one image/segmentation pair to tf example.

    Args:
      image: string of image data.
      filename: image filename.
      height: image height.
      width: image width.
      label_data: string of semantic segmentation data.

    Returns:
      tf example of one image/segmentation pair.
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'image/filename': _bytes_feature(filename),
        'objects/bbox':
            tf.io.FixedLenSequenceFeature((4,), tf.float32, allow_missing=True),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(3),
        'image/segmentation/class/encoded': _bytes_feature(label_data),
        'image/segmentation/class/format': _bytes_feature(label_format),
    }))