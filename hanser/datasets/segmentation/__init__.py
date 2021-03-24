import tensorflow as tf
from hanser.datasets.tfrecord import parse_tfexample_to_img_seg

def decode(example_proto):
    example = parse_tfexample_to_img_seg(example_proto)
    image = tf.image.decode_image(example['image/encoded'])
    label = tf.image.decode_image(example['image/segmentation/class/encoded'])
    height, width = example['image/height'], example['image/width']
    image.set_shape([height, width, 3])
    label.set_shape([height, width, 1])
    return image, label