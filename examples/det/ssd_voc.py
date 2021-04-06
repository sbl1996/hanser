from hanser.datasets import prepare
from hanser.transform import color_jitter
from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds
from hanser.tpu import setup, distribute_datasets

from hanser.detection import match_anchors
from hanser.detection.anchor import SSDAnchorGenerator
from hanser.transform.detection import random_expand, random_sample_crop, pad_to_fixed_size, random_hflip

from hanser.models.layers import set_defaults

output_size = 300

anchor_gen = SSDAnchorGenerator(
    strides=[8, 16, 32, 64, 100, 300],
    ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    basesize_ratio_range=(0.2, 0.9),
    input_size=300,
)
featmap_sizes = [
    [38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1],
    # [64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2], [1, 1],
]
anchors = anchor_gen.grid_anchors(featmap_sizes)

@curry
def preprocess(d, output_size=output_size, max_objects=100, training=True):
    mean_rgb = tf.convert_to_tensor([123.68, 116.779, 103.939], tf.float32)
    std_rgb = tf.convert_to_tensor([58.393, 57.12, 57.375], tf.float32)

    image_id = d['image/filename']
    str_len = tf.strings.length(image_id)
    image_id = tf.strings.to_number(
        tf.strings.substr(image_id, str_len - 10, 6),
        out_type=tf.int32
    )
    image_id = tf.where(str_len == 10, image_id + 10000, image_id)

    image = tf.cast(d['image'], tf.float32)
    bboxes, labels, is_difficults = d['objects']['bbox'], d['objects']['label'] + 1, d['objects']['is_difficult']
    labels = tf.cast(labels, tf.int32)

    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    bboxes = bboxes * tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)[:, None]

    if training:
        image = color_jitter(image,
                             brightness=0.5,
                             contrast=0.5,
                             saturation=0.5,
                             hue=0.33)
        if tf.random.normal(()) < 0.5:
            image, bboxes = expand(image, bboxes, 4.0, mean_rgb)
        if tf.random.normal(()) < 0.5:
            image, bboxes, classes, is_difficults = random_sample_crop(
                image, bboxes, labels, is_difficults)
        image, bboxes = random_hflip(image, bboxes, 0.5)

    image = tf.image.resize(image, (output_size, output_size))

    loc_t, cls_t, n_pos, ignore = match_anchors(
        bboxes, labels, anchors, pos_iou_thr=0.5, neg_iou_thr=0.5)

    image = (image - mean_rgb) / std_rgb

    bboxes = pad_to_fixed_size(bboxes, 0, [max_objects, 4])
    labels = pad_to_fixed_size(labels, 0, [max_objects])
    is_difficults = pad_to_fixed_size(is_difficults, 0, [max_objects])

    return image, {'loc_t': loc_t, 'cls_t': cls_t, 'n_pos': n_pos,
                   'bbox': bboxes, 'label': labels, 'ignore': ignore,
                   'image_id': image_id, 'is_difficult': is_difficults}


mul = 1
n_train, n_val = 10582, 1449
batch_size, eval_batch_size = 16 * mul, 20 * 4
steps_per_epoch, val_steps = n_train // batch_size, n_val // eval_batch_size

ds_train = tfds.load("voc/2007", split="train", data_dir="/Users/hrvvi/Downloads/datasets/tfrecords",
               shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds_val = tfds.load("voc/2007", split="validation", data_dir="/Users/hrvvi/Downloads/datasets/tfrecords",
               shuffle_files=False, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
ds_train = prepare(ds_train, batch_size, preprocess(training=True),
                   training=True, repeat=False)
ds_val = prepare(ds_val, eval_batch_size, preprocess(training=False),
                 training=False, repeat=False, drop_remainder=True)
ds_train_dist, ds_val_dist = setup([ds_train, ds_val], fp16=True)

set_defaults({
    'bn': {
        'sync': True,
    }
})

