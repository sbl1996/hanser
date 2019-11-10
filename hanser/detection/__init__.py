import tensorflow as tf
from hanser.losses import focal_loss
from hanser.ops import index_put, to_float
from toolz import curry


def generate_mlvl_anchors(grid_sizes, anchor_sizes):
    mlvl_anchors = []
    for (ly, lx), sizes in zip(grid_sizes, anchor_sizes):
        n = tf.shape(sizes)[0]
        cy = (tf.tile(tf.reshape(tf.range(ly, dtype=tf.float32), [ly, 1, 1, 1]), [1, lx, n, 1]) + 0.5) / ly
        cx = (tf.tile(tf.reshape(tf.range(lx, dtype=tf.float32), [1, lx, 1, 1]), [ly, 1, n, 1]) + 0.5) / lx
        size = tf.tile(tf.reshape(sizes, [1, 1, n, 2]), [ly, lx, 1, 1])
        anchors = tf.concat([cy, cx, size], 3)
        anchors = tf.clip_by_value(anchors, 0.0, 1.0)
        mlvl_anchors.append(anchors)
    return mlvl_anchors


def tlbr2tlwh(boxes):
    tl = boxes[:, :2]
    br = boxes[:, 2:]
    hw = br - tl
    return tf.concat([tl, hw], axis=1)


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
    boxes_yx = (boxes[:, :2] + boxes[:, 2:]) / 2
    boxes_hw = boxes[:, 2:] - boxes[:, :2]
    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]
    tytx = (boxes_yx - anchors_yx) / anchors_hw
    thtw = tf.math.log(boxes_hw / anchors_hw)
    return tf.concat([tytx, thtw], axis=1)


def match_anchors(boxes, classes, anchors, pos_thresh=0.5):
    num_objects = tf.shape(boxes)[0]
    num_anchors = tf.shape(anchors)[0]
    ious = iou(boxes, anchors)

    # max_ious = tf.reduce_max(ious, axis=1)
    max_indices = tf.argmax(ious, axis=1, output_type=tf.int32)
    match_ious = tf.reduce_max(ious, axis=0)
    matches = tf.argmax(ious, axis=0, output_type=tf.int32)

    # match_ious0 = max_indices
    match_ious = index_put(match_ious, max_indices, 1)
    matches = index_put(matches, max_indices, tf.range(num_objects))
    pos = match_ious > pos_thresh
    # print(nonzero(pos))
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
def detection_loss(labels, preds, alpha=0.25, gamma=2.0):
    loc_t = labels['loc_t']
    cls_t = labels['cls_t']
    n_pos = labels['n_pos']
    total_pos = tf.reduce_sum(n_pos)

    loc_p = preds['loc_p']
    cls_p = preds['cls_p']

    num_classes = tf.shape(cls_p)[-1]
    cls_t = tf.one_hot(cls_t, num_classes)
    normalizer = to_float(total_pos)
    weights = to_float(loc_t != 0)
    loc_loss = smooth_l1_loss(loc_t, loc_p, weights) / normalizer
#     loc_loss = tf.compat.v1.losses.huber_loss(loc_t, loc_p, weights, reduction='weighted_sum') / normalizer
    cls_loss = focal_loss(cls_p, cls_t, alpha, gamma) / normalizer
    return loc_loss + cls_loss
