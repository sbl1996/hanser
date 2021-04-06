import tensorflow as tf

def decode(example):
    image_id = example['image/filename']
    str_len = tf.strings.length(image_id)
    image_id = tf.strings.to_number(
        tf.strings.substr(image_id, str_len - 10, 6),
        out_type=tf.int32
    )
    image_id = tf.where(str_len == 10, image_id + 10000, image_id)

    image = tf.cast(example['image'], tf.float32)
    objects = example['objects']
    bboxes, labels, is_difficults = objects['bbox'], objects['label'] + 1, objects['is_difficult']
    labels = tf.cast(labels, tf.int32)
    return image, bboxes, labels, is_difficults, image_id