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
    objects = {
        "bbox": example['objects']['bbox'],
        'label': tf.cast(example['objects']['label'] + 1, tf.int32),
        'is_difficult': example['objects']['is_difficult']
    }
    return image, objects, image_id