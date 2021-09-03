from toolz import curry, get

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.detection import postprocess, coords_to_absolute, BBoxCoder, \
    AnchorGenerator, DetectionLoss, iou_loss, focal_loss
from hanser.detection.assign import atss_match

from hanser.datasets.detection.voc import decode, make_voc_dataset_sub

from hanser.transform import normalize
from hanser.transform.detection import random_hflip, random_resize, resize, pad_to, random_crop, \
    pad_objects

HEIGHT = WIDTH = 256

anchor_gen = AnchorGenerator(
    strides=[8, 16, 32, 64, 128],
    ratios=[1.0],
    octave_base_scale=2,
    scales_per_octave=1,
)
featmap_sizes = [
    [32, 32], [16, 16], [8, 8], [4, 4], [2, 2],
]

mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes)
num_level_bboxes = [a.shape[0] for a in mlvl_anchors]
anchors = tf.concat(mlvl_anchors, axis=0)

bbox_coder = BBoxCoder(anchors, (0.1, 0.1, 0.2, 0.2))

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=50, training=True):
    image, objects, image_id = decode(example)

    # if training:
    #     image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
    #     image, objects = random_crop(image, objects, output_size)
    #     image, objects = random_hflip(image, objects, 0.5)
    # else:
    image = resize(image, output_size)

    image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    image, objects = pad_to(image, objects, output_size)

    gt_bboxes, gt_labels = get(['gt_bbox', 'gt_label'], objects)
    gt_bboxes = coords_to_absolute(gt_bboxes, tf.shape(image)[:2])
    objects = {'gt_bbox': gt_bboxes, **objects}

    bbox_targets, labels, centerness = atss_match(
        gt_bboxes, gt_labels, anchors, num_level_bboxes, topk=9, centerness=True)

    objects = pad_objects(objects, max_objects)

    # image = tf.cast(image, tf.bfloat16)
    return image, {'bbox_target': bbox_targets, 'label': labels, 'centerness': centerness,
                   **objects, 'image_id': image_id}


mul = 1
n_train, n_val = 16, 8
batch_size, eval_batch_size = 4 * mul, 4

output_size = (HEIGHT, WIDTH)
max_objects = 50

ds_train = tfds.load("voc/2012", split=f"train[:{n_train}]",
                     shuffle_files=True, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))

it = iter(ds_train)

example = next(it)

image, objects, image_id = decode(example)

# if training:
#     image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
#     image, objects = random_crop(image, objects, output_size)
#     image, objects = random_hflip(image, objects, 0.5)
# else:
image = resize(image, output_size)

image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
image, objects = pad_to(image, objects, output_size)

gt_bboxes, gt_labels = get(['gt_bbox', 'gt_label'], objects)
gt_bboxes = coords_to_absolute(gt_bboxes, tf.shape(image)[:2])
objects = {**objects, 'gt_bbox': gt_bboxes}

bbox_targets, labels, centerness = atss_match(
    gt_bboxes, gt_labels, anchors, num_level_bboxes, topk=9, centerness=True)

objects = pad_objects(objects, max_objects)
