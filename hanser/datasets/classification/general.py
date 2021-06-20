import tensorflow as tf

def decode(example):
    image = tf.cast(example['image'], tf.float32)
    label = tf.cast(example['label'], dtype=tf.int32)
    return image, label
