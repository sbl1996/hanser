from toolz import curry

import tensorflow as tf
import tensorflow_datasets as tfds
from hanser.tpu import setup, distribute_datasets

from hanser.detection import match_anchors, generate_mlvl_anchors
from hanser.transform.detection import expand, random_sample_crop, pad_to_fixed_size

output_size = 300
priors = [
    [[30., 30.],
     [42.426407, 42.426407],
     [42.426407, 21.213203],
     [21.213203, 42.426407]],

    [[60., 60.],
     [82.15839, 82.15839],
     [84.85281, 42.426407],
     [42.426407, 84.85281],
     [103.92305, 34.641018],
     [34.641018, 103.92305]],

    [[112.5, 112.5],
     [136.24426, 136.24426],
     [159.09903, 79.549515],
     [79.549515, 159.09903],
     [194.85571, 64.951904],
     [64.951904, 194.85571]],

    [[165., 165.],
     [189.43996, 189.43996],
     [233.34525, 116.67262],
     [116.67262, 233.34525],
     [285.7884, 95.262794],
     [95.262794, 285.7884]],

    [[217.5, 217.5],
     [242.33241, 242.33241],
     [307.59146, 153.79573],
     [153.79573, 307.59146]],

    [[270., 270.],
     [284.60498, 284.60498],
     [381.83768, 190.91884],
     [190.91884, 381.83768]],
]
priors = [
    tf.convert_to_tensor(p) / output_size
    for p in priors
]
grid_sizes = [
    [38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]
    #     [64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2], [1, 1]
]
anchors = generate_mlvl_anchors(grid_sizes, priors)


@curry
def preprocess(d, output_size=output_size, pos_thresh=0.5, max_objects=100, training=True):
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
    bboxes, classes, is_difficults = d['objects']['bbox'], d['objects']['label'] + 1, d['objects']['is_difficult']
    classes = tf.cast(classes, tf.int32)

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
                image, bboxes, classes, is_difficults)
        # image, bboxes = random_hflip(image, bboxes, 0.5)

    image = tf.image.resize(image, (output_size, output_size))

    loc_t, cls_t, n_pos = match_anchors(bboxes, classes, anchors, pos_thresh)

    image = (image - mean_rgb) / std_rgb

    boxes = pad_to_fixed_size(bboxes, 0, [max_objects, 4])
    classes = pad_to_fixed_size(classes, 0, [max_objects])
    is_difficults = pad_to_fixed_size(is_difficults, 0, [max_objects])

    return image, {'loc_t': loc_t, 'cls_t': cls_t, 'n_pos': n_pos,
                   'bbox': boxes, 'label': classes, 'image_id': image_id,
                   'is_difficult': is_difficults}


ds = tfds.load("voc/2007", split="train+validation", data_dir="gs://hrvvi-datasets/tfds")
ds1 = ds.map(preprocess)
