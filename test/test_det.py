from glob import glob

import numpy as np
import tensorflow as tf

from hanser.datasets.tfrecord import parse_voc_example
from hanser.transform.detection import get_random_scale, resize_and_crop_image, resize_and_crop_boxes, draw_bboxes, \
    VOC_CATEGORIES, resize_with_pad

from hanser.detection import tlbr2tlwh, iou, generate_mlvl_anchors, match_anchors


def get_anchors():
    return tf.convert_to_tensor([
        [[22.6274, 45.2548],
         [32.0000, 64.0000],
         [32.0000, 32.0000],
         [45.2548, 45.2548],
         [45.2548, 22.6274],
         [64.0000, 32.0000]],

        [[45.2548, 90.5097],
         [64.0000, 128.0000],
         [64.0000, 64.0000],
         [90.5097, 90.5097],
         [90.5097, 45.2548],
         [128.0000, 64.0000]],

        [[90.5097, 181.0193],
         [128.0000, 256.0000],
         [128.0000, 128.0000],
         [181.0193, 181.0193],
         [181.0193, 90.5097],
         [256.0000, 128.0000]],

        [[181.0193, 362.0387],
         [256.0000, 512.0000],
         [256.0000, 256.0000],
         [362.0387, 362.0387],
         [362.0387, 181.0193],
         [512.0000, 256.0000]]])



def test_resize_and_crop():
    files = glob('/Users/hrvvi/tensorflow_datasets/voc/2007/4.0.0/voc-train.*')
    ds = tf.data.TFRecordDataset(files).map(parse_voc_example)
    it = iter(ds)
    d = next(it)
    image = d['image']
    height, width = image.shape[:2]
    boxes, classes = d['objects/bbox'], d['objects/label']
    output_size = 320
    # img_scale, scaled_height, scaled_width, offset_x, offset_y = get_random_scale(height, width, output_size, 1, 4)
    # image = resize_and_crop_image(image, scaled_height, scaled_width, output_size, offset_x, offset_y)
    # boxes, classes = resize_and_crop_boxes(boxes, classes, scaled_height, scaled_width, output_size, offset_x, offset_y)
    # image = tf.image.resize_with_pad(image, output_size, output_size)
    image, boxes = resize_with_pad(image, boxes, output_size, output_size)
    boxes1 = boxes * output_size
    draw_bboxes(image.numpy().astype(np.uint8), boxes1.numpy(), classes.numpy(), VOC_CATEGORIES[1:])


def test_iou():
    m = 1000
    n = 100
    boxes1 = tlbr2tlwh(tf.random.normal([m, 4]))
    boxes2 = tlbr2tlwh(tf.random.normal([n, 4]))
    ious = iou(boxes1, boxes2).numpy()
    from horch._numpy import iou_mn
    expected = iou_mn(boxes1.numpy(), boxes2.numpy())
    np.testing.assert_allclose(ious, expected)


def test_anchor_match():
    files = glob('/Users/hrvvi/tensorflow_datasets/voc/2007/4.0.0/voc-train.*')
    ds = tf.data.TFRecordDataset(files).map(parse_voc_example)
    it = iter(ds)
    d = next(it)
    image = d['image']
    height, width = image.shape[:2]
    boxes, classes = d['objects/bbox'], d['objects/label'] + 1

    output_size = 320
    image, boxes = resize_with_pad(image, boxes, output_size, output_size)

    anchors = get_anchors() / output_size
    grid_sizes = [
        [40, 40], [20, 20], [10, 10], [5, 5]
    ]
    mlvl_anchors = generate_mlvl_anchors(grid_sizes, anchors)
    anchors = tf.concat([tf.reshape(a, [-1, 4]) for a in mlvl_anchors], axis=0)
    loc_t, cls_t, ignore = match_anchors(boxes, classes, anchors)
    print(len(cls_t[cls_t != 0]) / len(boxes))