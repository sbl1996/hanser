import colorsys
import random

import numpy as np
import tensorflow as tf

from hanser.ops import index_put, get_shape
from hanser.detection.anchor import AnchorGenerator
from hanser.detection.assign import max_iou_assign, atss_assign
from hanser.detection.nms import batched_nms
from hanser.detection.iou import bbox_iou
from hanser.detection.bbox import BBoxCoder, coords_to_absolute
from hanser.detection.loss import DetectionLoss, focal_loss, iou_loss, l1_loss, smooth_l1_loss, cross_entropy_det

def encode_target(gt_bboxes, gt_labels, assigned_gt_inds,
                  bbox_coder: BBoxCoder = None, encode_bbox=True,
                  centerness=False):
    if bbox_coder is not None:
        assert encode_bbox or centerness
    num_gts = get_shape(gt_bboxes, 0)
    num_anchors = get_shape(assigned_gt_inds, 0)
    bbox_targets = tf.zeros([num_anchors, 4], dtype=tf.float32)
    labels = tf.zeros([num_anchors,], dtype=tf.int32)
    centerness_t = tf.zeros([num_anchors,], dtype=tf.float32)

    if num_gts == 0:
        ignore = tf.fill([num_anchors,], False)
        if centerness:
            return bbox_targets, labels, centerness_t, ignore
        return bbox_targets, labels, ignore

    pos = assigned_gt_inds > 0
    ignore = assigned_gt_inds == -1
    indices = tf.range(num_anchors, dtype=tf.int32)[pos]

    assigned_gt_inds = tf.gather(assigned_gt_inds, indices) - 1

    assigned_gt_bboxes = tf.gather(gt_bboxes, assigned_gt_inds)
    assigned_gt_labels = tf.gather(gt_labels, assigned_gt_inds)

    if bbox_coder:
        assigned_anchors = tf.gather(bbox_coder.anchors, indices)
        if centerness:
            assigned_centerness = bbox_coder.centerness(assigned_gt_bboxes, assigned_anchors)
            centerness_t = index_put(centerness_t, indices, assigned_centerness)
        if encode_bbox:
            assigned_gt_bboxes = bbox_coder.encode(assigned_gt_bboxes, assigned_anchors)

    bbox_targets = index_put(bbox_targets, indices, assigned_gt_bboxes)
    labels = index_put(labels, indices, assigned_gt_labels)

    if centerness:
        return bbox_targets, labels, centerness_t, ignore
    return bbox_targets, labels, ignore


def postprocess(bbox_preds, cls_scores, bbox_coder, centerness=None,
                nms_pre=5000, iou_threshold=0.5, score_threshold=0.05,
                topk=200, soft_nms_sigma=0., use_sigmoid=False, label_offset=0):
    if use_sigmoid:
        scores = tf.sigmoid(cls_scores)
    else:
        scores = tf.math.softmax(cls_scores, -1)
    scores = scores[..., label_offset:]
    if centerness is not None:
        scores = scores * tf.sigmoid(centerness)[..., None]

    anchors = bbox_coder.anchors
    if nms_pre < anchors.shape[0]:
        max_scores = tf.reduce_max(scores, axis=-1)
        idx = tf.math.top_k(max_scores, nms_pre, sorted=False)[1]
        bbox_preds = tf.gather(bbox_preds, idx, axis=1, batch_dims=1)
        scores = tf.gather(scores, idx, axis=1, batch_dims=1)
        anchors = tf.gather(anchors, idx, axis=0, batch_dims=0)

    bboxes = bbox_coder.decode(bbox_preds, anchors)

    bboxes = tf.expand_dims(bboxes, 2)
    bboxes, scores, labels, n_valids = batched_nms(
        bboxes, scores, iou_threshold, score_threshold,
        soft_nms_sigma=soft_nms_sigma, max_per_class=100, topk=topk)

    return {
        'bbox': bboxes,
        'score': scores,
        'label': labels,
        'n_valid': n_valids,
    }


def draw_bboxes(img, anns, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16, figsize=(10, 10)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not colors:
        if categories:
            colors = random_colors(len(categories))
        else:
            colors = ['w' for _ in range(100)]

    height, width = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for ann in anns:
        box = (ann['bbox'].reshape(2, 2) * [height, width])[:, ::-1].reshape(-1)
        box[2:] -= box[:2]
        cls = ann['category_id']
        color = colors[cls]
        rect = Rectangle(box[:2], box[2], box[3], linewidth=linewidth,
                         alpha=0.7, edgecolor=color, facecolor='none')

        ax.add_patch(rect)
        if categories:
            text = "%s %.2f" % (categories[cls], ann['score'])
            ax.text(box[0], box[1] + label_offset, text,
                    color=color, size=fontsize, backgroundcolor="none")
    return fig, ax


def draw_bboxes2(img, boxes, classes=None, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16,
                 figsize=(10, 10)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not colors:
        if categories:
            colors = random_colors(len(categories))
        else:
            colors = ['w' for _ in range(100)]

    # boxes = boxes.reshape(-1, 2, 2)[..., ::-1].reshape(-1, 4)
    boxes = (boxes.reshape(-1, 2, 2) * np.array(img.shape[:2]))[..., ::-1].reshape(-1, 4)
    boxes[:, 2:] -= boxes[:, :2]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for box, cls in zip(boxes, classes):
        color = colors[cls]
        rect = Rectangle(box[:2], box[2], box[3], linewidth=linewidth,
                         alpha=0.7, edgecolor=color, facecolor='none')

        ax.add_patch(rect)
        if categories:
            text = "%s" % categories[cls]
            # text = "%s %.2f" % (categories[cls], ann['score'])
            ax.text(box[0], box[1] + label_offset, text,
                    color=color, size=fontsize, backgroundcolor="none")
    return fig, ax


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def random_bboxes(shape):
    yx = tf.random.uniform(tuple(shape) + (2,))
    hw_half = tf.random.uniform(tuple(shape) + (2,), 0, 0.5)
    bboxes = tf.stack([yx - hw_half, yx + hw_half], axis=-2)
    bboxes = tf.reshape(bboxes, tuple(shape) + (4,))
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return bboxes