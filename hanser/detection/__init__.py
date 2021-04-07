import colorsys
import random

from hanser.detection.assign import max_iou_assign
from toolz import curry

import numpy as np
import tensorflow as tf
from hanser.losses import focal_loss
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
    loc_t = tf.concat([tytx, thtw], axis=1)
    std = tf.constant(std, dtype=loc_t.dtype)
    return loc_t / std


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
    loc_t = tf.zeros([num_anchors, 4], dtype=tf.float32)
    cls_t = tf.zeros([num_anchors,], dtype=tf.int32)
    if num_gts == 0:
        pos = tf.fill([num_anchors,], False)
        ignore = tf.fill([num_anchors,], False)
        return loc_t, cls_t, pos, ignore
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
    loc_t = index_put(loc_t, indices, assigned_bboxes)
    cls_t = index_put(cls_t, indices, assigned_gt_labels)

    return loc_t, cls_t, pos, ignore


def smooth_l1_loss(labels, preds, beta=1.0):
    abs_error = tf.math.abs(preds - labels)
    losses = tf.where(
        abs_error < beta,
        0.5 * abs_error * abs_error / beta,
        abs_error - 0.5 * beta
    )
    return losses


def l1_loss(labels, preds):
    losses = tf.math.abs(preds - labels)
    return losses


@curry
def detection_loss(target, preds, loc_loss='l1', cls_loss='focal', neg_pos_ratio=None,
                   alpha=0.25, gamma=2.0, label_smoothing=0, loc_loss_weight=1.):
    loc_t = target['loc_t']
    cls_t = target['cls_t']
    pos = target['pos']
    ignore = target['ignore']
    pos = tf.cast(pos, tf.float32)
    n_pos = tf.reduce_sum(pos, axis=1)

    loc_p = preds['loc_p']
    cls_p = preds['cls_p']

    total_pos = tf.reduce_sum(n_pos) + 1

    if loc_loss == 'l1':
        loc_losses = l1_loss(loc_t, loc_p)
    elif loc_loss == 'smooth_l1':
        loc_losses = smooth_l1_loss(loc_t, loc_p, beta=1.0)
    else:
        raise ValueError("Not supported regression loss: %s" % loc_loss)
    loc_loss = tf.reduce_sum(loc_losses * pos[:, :, None]) / total_pos

    if cls_loss == 'focal':
        num_classes = tf.shape(cls_p)[-1]
        cls_t = tf.one_hot(cls_t, num_classes + 1)
        cls_losses = focal_loss(cls_t[..., 1:], cls_p, alpha, gamma, label_smoothing)
        weight = tf.cast(~ignore, tf.float32)[:, :, None]
        cls_loss = tf.reduce_sum(cls_losses * weight) / total_pos
    elif cls_loss == 'ce':
        cls_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(cls_t, cls_p)
        if not neg_pos_ratio:
            cls_loss = tf.reduce_sum(cls_losses) / total_pos
        else:
            neg = tf.cast((~pos) & (~ignore), tf.float32)
            cls_loss_pos = tf.reduce_sum(cls_losses * pos)
            cls_loss_neg = hard_negative_mining(cls_losses * neg, n_pos, neg_pos_ratio)
            cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos
    else:
        raise ValueError("Not supported classification loss: %s" % cls_loss)

    return loc_loss * loc_loss_weight + cls_loss


def hard_negative_mining(losses, n_pos, neg_pos_ratio, max_pos=1000):
    # shape = tf.shape(losses)
    # batch_size = shape[0]
    # ind = tf.tile(tf.range(max_pos, dtype=tf.int32)[None], [batch_size, 1])
    ind = tf.range(max_pos, dtype=tf.int32)[None, :]
    weights = tf.cast(ind < n_pos[:, None] * neg_pos_ratio, tf.float32)
    losses = tf.math.top_k(losses, k=max_pos, sorted=True)[0]
    return tf.reduce_sum(weights * losses)



def detect(loc_p, cls_p, anchors, iou_threshold=0.5, conf_threshold=0.1, topk=100):
    logits = cls_p[:, 1:]
    scores = tf.sigmoid(tf.reduce_max(logits, axis=1))
    classes = tf.argmax(logits, axis=1, output_type=tf.int32) + 1

    mask = scores > conf_threshold
    loc_p = loc_p[mask]
    scores = scores[mask]
    classes = classes[mask]
    anchors = anchors[mask]

    bboxes = bbox_decode(loc_p, anchors)
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


def batched_detect(loc_p, cls_p, anchors, iou_threshold=0.5,
                   conf_threshold=0.05, topk=200, conf_strategy='softmax',
                   bbox_std=(1., 1., 1., 1.)):
    if conf_strategy == 'sigmoid':
        scores = tf.sigmoid(cls_p)
    else:
        scores = tf.math.softmax(cls_p, -1)
    bboxes = bbox_decode(loc_p, anchors, bbox_std)
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