import colorsys
import random

import numpy as np
import tensorflow as tf

from hanser.detection.anchor import AnchorGenerator, YOLOAnchorGenerator
from hanser.detection.assign import max_iou_match, atss_match, fcos_match, grid_points, yolo_match
from hanser.detection.nms import batched_nms
from hanser.detection.iou import bbox_iou2
from hanser.detection.bbox import BBoxCoder, FCOSBBoxCoder, coords_to_absolute, YOLOBBoxCoder
from hanser.detection.loss import DetectionLoss, focal_loss, iou_loss, l1_loss, smooth_l1_loss, cross_entropy_det, GFLoss, GFLossV2


def postprocess(bbox_preds, cls_scores, bbox_coder, centerness=None,
                nms_pre=5000, iou_threshold=0.5, score_threshold=0.05,
                topk=200, soft_nms_sigma=0., use_sigmoid=False, label_offset=0,
                from_logits=True):
    if from_logits:
        if use_sigmoid:
            scores = tf.sigmoid(cls_scores)
        else:
            scores = tf.math.softmax(cls_scores, -1)
    else:
        scores = cls_scores
    scores = scores[..., label_offset:]
    if centerness is not None:
        scores = tf.sqrt(scores * tf.sigmoid(centerness)[..., None])

    num_dets = bbox_preds.shape[1]
    if nms_pre < num_dets:
        max_scores = tf.reduce_max(scores, axis=-1)
        idx = tf.math.top_k(max_scores, nms_pre, sorted=False)[1]
        bbox_preds = tf.gather(bbox_preds, idx, axis=1, batch_dims=1)
        scores = tf.gather(scores, idx, axis=1, batch_dims=1)
        bboxes = bbox_coder.decode(bbox_preds, idx)
    else:
        bboxes = bbox_coder.decode(bbox_preds)

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


def draw_bboxes2(img, boxes, classes, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16,
                 figsize=(10, 10), relative=True):
    # boxes is (x1, y1, x2, y2) and in [0, 1]
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not colors:
        if categories:
            colors = random_colors(len(categories))
        else:
            colors = ['w' for _ in range(100)]

    # boxes = boxes.reshape(-1, 2, 2)[..., ::-1].reshape(-1, 4)
    boxes = boxes.copy()
    if relative:
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