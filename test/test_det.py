from glob import glob

import numpy as np
import tensorflow as tf

from hanser.datasets.tfrecord import parse_voc_example
from hanser.transform.detection import get_random_scale, resize_and_crop_image, resize_and_crop_boxes, resize_with_pad, \
    random_sample_crop, random_apply, random_hflip, expand
from hanser.datasets.voc import VOC_CATEGORIES

from hanser.detection import tlbr2tlhw, iou, generate_mlvl_anchors, match_anchors, draw_bboxes2, yxhw2tlbr
from hanser.detection import iou_mn as iou1


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
    boxes1 = tlbr2tlhw(tf.random.normal([m, 4]))
    boxes2 = tlbr2tlhw(tf.random.normal([n, 4]))
    # ious = iou(boxes1, boxes2).numpy()
    ious = iou1(boxes1.numpy(), boxes2.numpy())
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
    image, boxes = resize_with_pad(image, boxes, output_size, output_size, 0)

    anchors = get_anchors() / output_size
    grid_sizes = [
        [40, 40], [20, 20], [10, 10], [5, 5]
    ]
    mlvl_anchors = generate_mlvl_anchors(grid_sizes, anchors)
    anchors = tf.concat([tf.reshape(a, [-1, 4]) for a in mlvl_anchors], axis=0)
    loc_t, cls_t, ignore = match_anchors(boxes, classes, anchors)
    print(len(cls_t[cls_t != 0]) / len(boxes))


def test_batched_detect():
    batch_size = 10
    num_anchors = 10000
    num_classes = 21
    anchors = tf.random.normal([num_anchors, 4])
    loc_p = tf.random.normal([batch_size, num_anchors, 4])
    cls_p = tf.random.normal([batch_size, num_anchors, num_classes])


def test_combined_nms():
    batch_size = 10
    num_boxes = 10000
    num_classes = 21
    boxes = tf.random.normal([batch_size, num_boxes, 4])
    boxes = yxhw2tlbr(boxes)
    scores = tf.random.uniform([batch_size, num_boxes, num_classes])
    max_output_size_per_class = 20
    max_total_size = 100
    score_threshold = 0.1
    boxes1 = tf.expand_dims(boxes, 2)
    tf.image.combined_non_max_suppression(boxes1, scores, 20, 100, 0.5, 0.5)
    # boxes, scores, classes, n_valids = tf.image.combined_non_max_suppression(boxes1, scores, 20, 100, 0.5, 0.1)



def test_transform():
    files = glob('/Users/hrvvi/tensorflow_datasets/voc/2007/4.0.0/voc-train.*')
    ds = tf.data.TFRecordDataset(files).map(parse_voc_example)
    it = iter(ds)

    d = next(it)
    image = tf.cast(d['image'], tf.float32)
    bboxes, classes, is_difficults = d['objects/bbox'], d['objects/label'] + 1, d['objects/is_difficult']
    classes = tf.cast(classes, tf.int32)

    mean_rgb = tf.convert_to_tensor([127.5, 127.5, 127.5], tf.float32)

    if tf.random.normal(()) < 0.5:
        image, bboxes = expand(image, bboxes, 4.0, mean_rgb)
    # if tf.random.normal(()) < 0.5:
    #     image, bboxes, classes, is_difficults = random_sample_crop(
    #         image, bboxes, classes, is_difficults)
    image, bboxes = random_hflip(image, bboxes, 0.5)
    output_size = 300
    image = tf.image.resize(image, (output_size, output_size))
    #
    # # image, bboxes = resize_with_pad(image, bboxes, output_size, output_size, mean_rgb)

    draw_bboxes2(image.numpy().astype(np.uint8), bboxes.numpy(), classes.numpy(), VOC_CATEGORIES)