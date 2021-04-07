import colorsys
import random

from hanser.detection.assign import max_iou_assign
from toolz import curry

import numpy as np
import tensorflow as tf
from hanser.losses import focal_loss, l1_loss, reduce_loss
from hanser.ops import index_put, to_float, get_shape


def coords_to_absolute(bboxes, size):
    height, width = size[0], size[1]
    bboxes = bboxes * tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)[None, :]
    return bboxes

def bbox_encode(bboxes, anchors, std=(1., 1., 1., 1.)):
    boxes_yx = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    boxes_hw = bboxes[:, 2:] - bboxes[:, :2]
    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    tytx = (boxes_yx - anchors_yx) / anchors_hw
    thtw = tf.math.log(boxes_hw / anchors_hw)
    box_t = tf.concat([tytx, thtw], axis=1)
    std = tf.constant(std, dtype=box_t.dtype)
    return box_t / std


def bbox_decode(pred, anchors, std=(1., 1., 1., 1.)):

    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]

    std = tf.constant(std, dtype=pred.dtype)
    pred = pred * std

    bbox_yx = (pred[..., :2] * anchors_hw) + anchors_yx
    bbox_hw = tf.exp(pred[..., 2:]) * anchors_hw
    bbox_hw_half = bbox_hw / 2
    bboxes = tf.concat([bbox_yx - bbox_hw_half, bbox_yx + bbox_hw_half], axis=-1)
    return bboxes


def match_anchors(gt_bboxes, gt_labels, anchors, pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=.0,
                  bbox_std=(1., 1., 1., 1.)):
    num_gts = tf.shape(gt_bboxes)[0]
    num_anchors = get_shape(anchors, 0)
    box_t = tf.zeros([num_anchors, 4], dtype=tf.float32)
    cls_t = tf.zeros([num_anchors,], dtype=tf.int32)

    if num_gts == 0:
        ignore = tf.fill([num_anchors,], False)
        return box_t, cls_t, ignore

    assigned_gt_inds = max_iou_assign(anchors, gt_bboxes, pos_iou_thr, neg_iou_thr, min_pos_iou,
                                      match_low_quality=True, gt_max_assign_all=False)

    pos = assigned_gt_inds > 0
    ignore = assigned_gt_inds == -1
    indices = tf.range(num_anchors, dtype=tf.int32)[pos]

    assigned_gt_inds = tf.gather(assigned_gt_inds, indices) - 1

    assigned_gt_bboxes = tf.gather(gt_bboxes, assigned_gt_inds)
    assigned_gt_labels = tf.gather(gt_labels, assigned_gt_inds)
    assigned_anchors = tf.gather(anchors, indices)

    assigned_bboxes = bbox_encode(assigned_gt_bboxes, assigned_anchors, bbox_std)
    box_t = index_put(box_t, indices, assigned_bboxes)
    cls_t = index_put(cls_t, indices, assigned_gt_labels)

    return box_t, cls_t, ignore


@curry
def detection_loss(target, preds, box_loss=None, cls_loss=None, box_loss_weight=1.):
    box_loss = box_loss or l1_loss
    cls_loss = cls_loss or focal_loss

    box_t = target['box_t']
    cls_t = target['cls_t']
    non_ignore = to_float(~target['ignore'])

    pos = to_float(cls_t != 0)
    total_pos = tf.reduce_sum(pos) + 1

    box_p = preds['box_p']
    cls_p = preds['cls_p']

    box_losses = box_loss(box_t, box_p, weight=pos[..., None], reduction='sum')
    box_loss = box_losses / total_pos

    cls_losses = cls_loss(cls_t, cls_p, weight=non_ignore, reduction='sum')
    cls_loss = cls_losses / total_pos
    return box_loss * box_loss_weight + cls_loss


def detect(box_p, cls_p, anchors, iou_threshold=0.5, conf_threshold=0.1, topk=100):
    logits = cls_p[:, 1:]
    scores = tf.sigmoid(tf.reduce_max(logits, axis=1))
    classes = tf.argmax(logits, axis=1, output_type=tf.int32) + 1

    mask = scores > conf_threshold
    box_p = box_p[mask]
    scores = scores[mask]
    classes = classes[mask]
    anchors = anchors[mask]

    bboxes = bbox_decode(box_p, anchors)
    indices = tf.image.non_max_suppression(bboxes, scores, topk, iou_threshold, conf_threshold)
    dets = []
    for i in indices:
        dets.append({
            'image_id': -1,
            'category_id': classes[i].numpy(),
            'bbox': bboxes[i].numpy(),
            'score': scores[i].numpy()
        })
    return dets


def batched_detect(box_p, cls_p, anchors, iou_threshold=0.5,
                   conf_threshold=0.05, topk=200, conf_strategy='softmax',
                   bbox_std=(1., 1., 1., 1.), label_offset=0):
    if conf_strategy == 'sigmoid':
        scores = tf.sigmoid(cls_p)
    else:
        scores = tf.math.softmax(cls_p, -1)
    if label_offset:
        scores = scores[..., label_offset:]
    bboxes = bbox_decode(box_p, anchors, bbox_std)
    bboxes = tf.expand_dims(bboxes, 2)
    bboxes, scores, labels, n_valids = tf.image.combined_non_max_suppression(
        bboxes, scores, 100, topk, iou_threshold, conf_threshold, clip_boxes=False)
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


class BBox:
    LTWH = 0  # [xmin, ymin, width, height]
    LTRB = 1  # [xmin, ymin, xmax,  ymax]
    XYWH = 2  # [cx,   cy,   width, height]

    def __init__(self, image_id, category_id, bbox, score=None, is_difficult=False, area=None, segmentation=None,
                 **kwargs):
        self.image_id = image_id
        self.category_id = category_id
        self.score = score
        self.is_difficult = is_difficult
        self.bbox = bbox
        self.area = area
        self.segmentation = segmentation

    def __repr__(self):
        return "BBox(image_id=%s, category_id=%s, bbox=%s, score=%s, is_difficult=%s, area=%s)" % (
            self.image_id, self.category_id, self.bbox, self.score, self.is_difficult, self.area
        )


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