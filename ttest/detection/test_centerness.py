from toolz import curry, get

import tensorflow as tf

from hanser.losses import focal_loss
from hanser.detection import DetectionLoss, postprocess, coords_to_absolute, BBoxCoder, iou_loss
from hanser.detection.assign import atss_assign, encode_target
from hanser.detection.anchor import AnchorGenerator

from hanser.datasets.detection.voc import decode, make_voc_dataset_sub

from hanser.transform import normalize
from hanser.transform.detection import pad_to_fixed_size, random_hflip, random_resize, resize, pad_to, random_crop

from hanser.train.metrics import MeanAveragePrecision


HEIGHT = WIDTH = 640

anchor_gen = AnchorGenerator(
    strides=[8, 16, 32, 64, 128],
    ratios=[1.0],
    octave_base_scale=6,
    scales_per_octave=1,
)
featmap_sizes = [
    [80, 80], [40, 40], [20, 20], [10, 10], [5, 5],
]

mlvl_anchors = anchor_gen.grid_anchors(featmap_sizes)
num_level_bboxes = [a.shape[0] for a in mlvl_anchors]
anchors = tf.concat(mlvl_anchors, axis=0)

bbox_coder = BBoxCoder(anchors)

@curry
def preprocess(example, output_size=(HEIGHT, WIDTH), max_objects=50, training=True):
    image, objects, image_id = decode(example)

    # if training:
        # image = random_resize(image, output_size, ratio_range=(0.8, 1.2))
        # image, objects = random_crop(image, objects, output_size)
        # image, objects = random_hflip(image, objects, 0.5)
    # else:
    image = resize(image, output_size)

    image = normalize(image, [123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    image, objects = pad_to(image, objects, output_size)

    gt_bboxes, gt_labels, is_difficults = get(['bbox', 'label', 'is_difficult'], objects)
    gt_bboxes = coords_to_absolute(gt_bboxes, tf.shape(image)[:2])

    assigned_gt_inds = atss_assign(anchors, num_level_bboxes, gt_bboxes, topk=9)
    bbox_targets, labels, centerness, ignore = encode_target(
        gt_bboxes, gt_labels, assigned_gt_inds, bbox_coder,
        encode_bbox=False, centerness=True)

    gt_bboxes = pad_to_fixed_size(gt_bboxes, max_objects)
    gt_labels = pad_to_fixed_size(gt_labels, max_objects)
    is_difficults = pad_to_fixed_size(is_difficults, max_objects)

    # image = tf.cast(image, tf.bfloat16)
    return image, {'bbox_target': bbox_targets, 'label': labels, 'ignore': ignore,
                   'centerness': centerness, 'gt_bbox': gt_bboxes, 'gt_label': gt_labels,
                   'is_difficult': is_difficults, 'image_id': image_id}

mul = 1
n_train, n_val = 128, 8
batch_size, eval_batch_size = 1, 4
ds_train, ds_val, steps_per_epoch, val_steps = make_voc_dataset_sub(
    n_train, n_val, batch_size, eval_batch_size, preprocess, prefetch=False)

it = iter(ds_train)

x, y = next(it)
pos = y['label'] != 0
cent = y['centerness'][pos]
print(cent)