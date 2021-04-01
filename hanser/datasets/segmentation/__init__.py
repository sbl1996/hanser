import tensorflow as tf
from hanser.datasets.segmentation.tfrecord import parse_tfexample_to_img_seg

def decode(example_proto):
    example = parse_tfexample_to_img_seg(example_proto)
    image = tf.image.decode_image(example['image/encoded'])
    label = tf.image.decode_image(example['image/segmentation/class/encoded'])
    image.set_shape([None, None, 3])
    label.set_shape([None, None, 1])
    return image, label
