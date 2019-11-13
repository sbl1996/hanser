import colorsys
import random

from toolz import curry

import numpy as np
import tensorflow as tf
from hanser.losses import focal_loss
from hanser.ops import index_put, to_float


def iou_mn(boxes1, boxes2):
    boxes2 = boxes2.T
    y_min1, x_min1, y_max1, x_max1 = np.split(boxes1, 4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = np.split(boxes2, 4, axis=0)
    y_min = np.maximum(y_min1, y_min2)
    x_min = np.maximum(x_min1, x_min2)
    y_max = np.minimum(y_max1, y_max2)
    x_max = np.minimum(x_max1, x_max2)
    inter_h = np.maximum(0.0, y_max - y_min)
    inter_w = np.maximum(0.0, x_max - x_min)
    inter_area = inter_h * inter_w
    areas1 = (y_max1 - y_min1) * (x_max1 - x_min1)
    areas2 = (y_max2 - y_min2) * (x_max2 - x_min2)
    union_area = areas1 + areas2 - inter_area
    return np.where(inter_area == 0, 0.0, inter_area / union_area)


def generate_mlvl_anchors(grid_sizes, anchor_sizes):
    mlvl_anchors = []
    for (ly, lx), sizes in zip(grid_sizes, anchor_sizes):
        n = tf.shape(sizes)[0]
        cy = (tf.tile(tf.reshape(tf.range(ly, dtype=tf.float32), [ly, 1, 1, 1]), [1, lx, n, 1]) + 0.5) / ly
        cx = (tf.tile(tf.reshape(tf.range(lx, dtype=tf.float32), [1, lx, 1, 1]), [ly, 1, n, 1]) + 0.5) / lx
        size = tf.tile(tf.reshape(sizes, [1, 1, n, 2]), [ly, lx, 1, 1])
        anchors = tf.concat([cy, cx, size], 3)
        anchors = yxhw2tlbr(anchors)
        mlvl_anchors.append(anchors)
    anchors = tf.concat([tf.reshape(a, [-1, 4]) for a in mlvl_anchors], axis=0)
    anchors = tf.clip_by_value(anchors, 0.0, 1.0)
    return anchors


def yxhw2tlbr(boxes):
    yx = boxes[..., :2]
    hw_half = boxes[..., 2:] / 2
    tl = yx - hw_half
    br = yx + hw_half
    return tf.concat([tl, br], axis=-1)


def tlbr2yxhw(boxes):
    tl = boxes[..., :2]
    br = boxes[..., 2:]
    yx = (tl + br) / 2
    hw = br - tl
    return tf.concat([yx, hw], axis=-1)


def tlbr2tlhw(boxes):
    tl = boxes[..., :2]
    br = boxes[..., 2:]
    hw = br - tl
    return tf.concat([tl, hw], axis=-1)


def iou(boxes1, boxes2):
    boxes2 = tf.transpose(boxes2)
    y_min1, x_min1, y_max1, x_max1 = tf.split(boxes1, 4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(boxes2, 4, axis=0)
    y_min = tf.maximum(y_min1, y_min2)
    x_min = tf.maximum(x_min1, x_min2)
    y_max = tf.minimum(y_max1, y_max2)
    x_max = tf.minimum(x_max1, x_max2)
    inter_h = tf.maximum(0.0, y_max - y_min)
    inter_w = tf.maximum(0.0, x_max - x_min)
    inter_area = inter_h * inter_w
    areas1 = (y_max1 - y_min1) * (x_max1 - x_min1)
    areas2 = (y_max2 - y_min2) * (x_max2 - x_min2)
    union_area = areas1 + areas2 - inter_area
    return tf.where(inter_area == 0, 0.0, inter_area / union_area)


def coords_to_target(boxes, anchors):
    # boxes: TLBR
    # anchors: TLBR
    boxes_yx = (boxes[:, :2] + boxes[:, 2:]) / 2
    boxes_hw = boxes[:, 2:] - boxes[:, :2]
    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    tytx = (boxes_yx - anchors_yx) / anchors_hw
    thtw = tf.math.log(boxes_hw / anchors_hw)
    return tf.concat([tytx * 10, thtw * 5], axis=1)


def target_to_coords(target, anchors):
    # boxes: YXHW (..., #anchors, 4)
    # anchors: TLBR (#anchors, 4)
    # return: TLBR

    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]

    boxes_yx = (target[..., :2] / 10 * anchors_hw) + anchors_yx
    boxes_hw = tf.exp(target[..., 2:] / 5) * anchors_hw
    boxes_hw_half = boxes_hw / 2
    boxes = tf.concat([boxes_yx - boxes_hw_half, boxes_yx + boxes_hw_half], axis=-1)
    return boxes


def match_anchors(boxes, classes, anchors, pos_thresh=0.5):
    num_objects = tf.shape(boxes)[0]
    num_anchors = tf.shape(anchors)[0]
    ious = iou(boxes, anchors)

    # max_ious = tf.reduce_max(ious, axis=1)
    max_indices = tf.argmax(ious, axis=1, output_type=tf.int32)
    match_ious = tf.reduce_max(ious, axis=0)
    matches = tf.argmax(ious, axis=0, output_type=tf.int32)

    # match_ious0 = match_ious
    match_ious = index_put(match_ious, max_indices, 1)
    matches = index_put(matches, max_indices, tf.range(num_objects))
    pos = match_ious > pos_thresh
    # print(match_ious0[pos])
    indices = tf.range(num_anchors, dtype=tf.int32)[pos]
    n_pos = tf.shape(indices)[0]

    matches = tf.gather(matches, indices)

    boxes = tf.gather(boxes, matches)
    classes = tf.gather(classes, matches)
    anchors = tf.gather(anchors, indices)

    loc_t = tf.zeros([num_anchors, 4], dtype=boxes.dtype)
    loc_t = index_put(loc_t, indices, coords_to_target(boxes, anchors))
    cls_t = tf.zeros([num_anchors, ], dtype=classes.dtype)
    cls_t = index_put(cls_t, indices, classes)

    # if neg_thresh:
    #     ignore = ~pos & (match_ious >= neg_thresh)
    # else:
    #     ignore = tf.zeros_like(cls_t, dtype=tf.bool)
    return loc_t, cls_t, n_pos


# huber loss
def smooth_l1_loss(labels, preds, weights, delta=1.0):
    error = preds - labels
    abs_error = tf.math.abs(error)
    quadratic = tf.math.minimum(abs_error, delta)
    linear = abs_error - quadratic
    losses = (quadratic * quadratic) * 0.5 + linear * delta
    losses = losses * weights
    return tf.reduce_sum(losses)


@curry
def detection_loss(labels, preds, cls_loss='focal', neg_pos_ratio=3, alpha=0.25, gamma=2.0, tpu=False):
    loc_t = labels['loc_t']
    cls_t = labels['cls_t']
    n_pos = labels['n_pos']

    loc_p = preds['loc_p']
    cls_p = preds['cls_p']

    if cls_loss == 'focal':
        delta = 1.0
        box_loss_weight = 10
    else:
        delta = 1.0
        box_loss_weight = 1.0

    total_pos = tf.reduce_sum(n_pos)
    normalizer = to_float(total_pos)
    weights = to_float(loc_t != 0)
    loc_loss = smooth_l1_loss(loc_t, loc_p, weights, delta=delta) / normalizer

    if cls_loss == 'focal':
        num_classes = tf.shape(cls_p)[-1]
        cls_t = tf.one_hot(cls_t, num_classes)
        cls_loss = focal_loss(cls_t, cls_p, alpha, gamma) / normalizer
    else:
        weights = tf.cast(tf.not_equal(cls_t, 0), cls_p.dtype)
        cls_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(cls_t, cls_p)
        cls_loss_pos = tf.reduce_sum(cls_losses * weights)
        if tpu:
            cls_loss_neg = tpu_hard_negative_mining(cls_losses * (1 - weights), n_pos, neg_pos_ratio)
        else:
            cls_loss_neg = hard_negative_mining(cls_losses * (1 - weights), n_pos, neg_pos_ratio)
        cls_loss = (cls_loss_pos + cls_loss_neg) / normalizer

    return box_loss_weight * loc_loss + cls_loss


def hard_negative_mining(losses, n_pos, neg_pos_ratio):

    def body(i, losses, n_pos, loss):
        n_neg = n_pos[i] * neg_pos_ratio
        hard_neg_losses = tf.math.top_k(losses[i], n_neg, sorted=False)[0]
        loss = loss + tf.reduce_sum(hard_neg_losses)
        return [i + 1, losses, n_pos, loss]

    loss = tf.while_loop(
        lambda i, losses, n_pos, loss: i < tf.shape(losses)[0],
        body,
        [0, losses, n_pos, 0.0]
    )[-1]
    return loss


def tpu_hard_negative_mining(losses, n_pos, neg_pos_ratio, max_pos=1000):
    shape = tf.shape(losses)
    batch_size = shape[0]
    # num_anchors = shape[1]
    ind = tf.tile(tf.range(max_pos, dtype=tf.int32)[None], [batch_size, 1])
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

    boxes = target_to_coords(loc_p, anchors)
    indices = tf.image.non_max_suppression(boxes, scores, topk, iou_threshold, conf_threshold)
    dets = []
    for i in indices:
        dets.append({
            'image_id': -1,
            'category_id': classes[i].numpy(),
            'bbox': boxes[i].numpy(),
            'score': scores[i].numpy()
        })
    return dets


def batched_detect(loc_p, cls_p, anchors, iou_threshold=0.5, conf_threshold=0.1, topk=200):
    scores = tf.math.softmax(cls_p, -1)[..., 1:]
    boxes = target_to_coords(loc_p, anchors)
    boxes = tf.expand_dims(boxes, 2)
    boxes, scores, classes, n_valids = tf.image.combined_non_max_suppression(
        boxes, scores, 100, topk, iou_threshold, conf_threshold)
    return {
        'bbox': boxes,
        'score': scores,
        'label': classes,
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
    shape = tf.TensorShape(shape)
    yxhw = tf.random.uniform(shape.concatenate(4))
    boxes = yxhw2tlbr(yxhw)
    boxes = tf.clip_by_value(boxes, 0, 1)
    return boxes
