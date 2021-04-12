import colorsys
import random

from toolz import curry

import numpy as np
import tensorflow as tf

from hanser.ops import index_put, to_float, get_shape
from hanser.detection.assign import max_iou_assign, atss_assign
from hanser.detection.nms import batched_nms
from hanser.losses import reduce_loss
from hanser.detection.iou import bbox_iou


class BBoxCoder:

    def __init__(self, anchors, bbox_std=(1., 1., 1., 1.)):
        self.anchors = tf.constant(anchors, tf.float32)
        self.bbox_std = tf.constant(bbox_std, tf.float32)

    def encode(self, bboxes, anchors=None):
        if anchors is None:
            anchors = self.anchors
        return bbox_encode(bboxes, anchors, self.bbox_std)

    def decode(self, bboxes, anchors=None):
        if anchors is None:
            anchors = self.anchors
        return bbox_decode(bboxes, anchors, self.bbox_std)

    def centerness(self, bboxes, anchors=None):
        if anchors is None:
            anchors = self.anchors
        return centerness_target(bboxes, anchors)


def coords_to_absolute(bboxes, size):
    height, width = size[0], size[1]
    bboxes = bboxes * tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)[None, :]
    return bboxes


def bbox_encode(bboxes, anchors, std=(1., 1., 1., 1.)):
    boxes_yx = (bboxes[..., :2] + bboxes[..., 2:]) / 2
    boxes_hw = bboxes[..., 2:] - bboxes[..., :2]
    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    tytx = (boxes_yx - anchors_yx) / anchors_hw
    boxes_hw = tf.maximum(boxes_hw, 1e-6)
    thtw = tf.math.log(boxes_hw / anchors_hw)
    box_t = tf.concat([tytx, thtw], axis=-1)
    std = tf.constant(std, dtype=box_t.dtype)
    return box_t / std


@curry
def bbox_decode(pred, anchors, std=(1., 1., 1., 1.)):
    anchors = tf.convert_to_tensor(anchors, pred.dtype)
    anchors_yx = (anchors[..., :2] + anchors[..., 2:]) / 2
    anchors_hw = anchors[..., 2:] - anchors[..., :2]

    std = tf.constant(std, dtype=pred.dtype)
    pred = pred * std

    bbox_yx = (pred[..., :2] * anchors_hw) + anchors_yx
    bbox_hw = tf.exp(pred[..., 2:]) * anchors_hw
    bbox_hw_half = bbox_hw / 2
    bboxes = tf.concat([bbox_yx - bbox_hw_half, bbox_yx + bbox_hw_half], axis=-1)
    return bboxes


def match_anchors(gt_bboxes, gt_labels, anchors, pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=.0):
    num_gts = get_shape(gt_bboxes, 0)
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

    box_t = index_put(box_t, indices, assigned_gt_bboxes)
    cls_t = index_put(cls_t, indices, assigned_gt_labels)

    return box_t, cls_t, ignore


def centerness_target(bboxes, anchors):
    anchors_cy = (anchors[:, 0] + anchors[:, 2]) / 2
    anchors_cx = (anchors[:, 1] + anchors[:, 3]) / 2
    t_ = anchors_cy - bboxes[..., 0]
    l_ = anchors_cx - bboxes[..., 1]
    b_ = bboxes[..., 2] - anchors_cy
    r_ = bboxes[..., 3] - anchors_cx

    centerness = tf.sqrt(
        (tf.minimum(l_, r_) / tf.maximum(l_, r_)) *
        (tf.minimum(t_, b_) / tf.maximum(t_, b_)))
    return centerness


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


class DetectionLoss:

    def __init__(self, box_loss_fn, cls_loss_fn, box_loss_weight=1.,
                 bbox_coder: BBoxCoder = None, decode_pred=False, centerness=False):
        if decode_pred or centerness:
            assert bbox_coder is not None
        self.box_loss_fn = box_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.box_loss_weight = box_loss_weight
        self.bbox_coder = bbox_coder
        self.decode_pred = decode_pred
        self.centerness = centerness

    def __call__(self, y_true, y_pred):

        bbox_targets = y_true['bbox_target']
        labels = y_true['label']
        non_ignore = to_float(~y_true['ignore'])

        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']

        pos = labels != 0
        pos_weight = to_float(pos)
        total_pos = tf.reduce_sum(pos_weight) + 1

        if self.decode_pred:
            dec_bbox_preds = self.bbox_coder.decode(bbox_preds)
            bbox_preds = dec_bbox_preds

        loss_box = self.box_loss_fn(bbox_targets, bbox_preds, weight=pos_weight, reduction='sum') / total_pos
        loss_cls = self.cls_loss_fn(labels, cls_scores, weight=non_ignore, reduction='sum') / total_pos

        loss = loss_box * self.box_loss_weight + loss_cls
        if self.centerness:
            centerness = y_pred['centerness']
            centerness_t = y_true['centerness']
            loss_centerness = tf.nn.sigmoid_cross_entropy_with_logits(
                centerness_t, centerness)
            loss_centerness = reduce_loss(loss_centerness, pos_weight, reduction='sum') / total_pos
            loss = loss + loss_centerness
        return loss


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


def random_bboxes(shape):
    yx = tf.random.uniform(tuple(shape) + (2,))
    hw_half = tf.random.uniform(tuple(shape) + (2,), 0, 0.5)
    bboxes = tf.stack([yx - hw_half, yx + hw_half], axis=-2)
    bboxes = tf.reshape(bboxes, tuple(shape) + (4,))
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    return bboxes


@curry
def iou_loss(y_true, y_pred, weight=None, mode='iou', reduction='sum'):
    # y_true: (batch_size, n_dts, 4)
    # y_pred: (batch_size, n_dts, 4)
    # weight: (batch_size, n_dts)
    losses = 1.0 - bbox_iou(y_true, y_pred, mode=mode, is_aligned=True)
    return reduce_loss(losses, weight, reduction)
