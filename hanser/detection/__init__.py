from toolz import curry

import numpy as np
import tensorflow as tf
from hanser.losses import focal_loss
from hanser.ops import index_put, to_float
from hanser.transform.detection import random_colors


def generate_mlvl_anchors(grid_sizes, anchor_sizes):
    mlvl_anchors = []
    for (ly, lx), sizes in zip(grid_sizes, anchor_sizes):
        n = tf.shape(sizes)[0]
        cy = (tf.tile(tf.reshape(tf.range(ly, dtype=tf.float32), [ly, 1, 1, 1]), [1, lx, n, 1]) + 0.5) / ly
        cx = (tf.tile(tf.reshape(tf.range(lx, dtype=tf.float32), [1, lx, 1, 1]), [ly, 1, n, 1]) + 0.5) / lx
        size = tf.tile(tf.reshape(sizes, [1, 1, n, 2]), [ly, lx, 1, 1])
        anchors = tf.concat([cy, cx, size], 3)
        anchors = yxhw2tlbr(anchors)
        anchors = tf.clip_by_value(anchors, 0.0, 1.0)
        mlvl_anchors.append(anchors)
    return mlvl_anchors


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
    return tf.concat([tytx, thtw], axis=1)


def target_to_coords(target, anchors):
    # boxes: YXHW
    # anchors: TLBR
    # return: TLBR

    anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
    anchors_hw = anchors[:, 2:] - anchors[:, :2]

    boxes_yx = (target[:, :2] * anchors_hw) + anchors_yx
    boxes_hw = tf.exp(target[:, 2:]) * anchors_hw
    boxes_hw_half = boxes_hw / 2
    boxes = tf.concat([boxes_yx - boxes_hw_half, boxes_yx + boxes_hw_half], axis=1)
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
    cls_loss = focal_loss(cls_t, cls_p, alpha, gamma) / normalizer
    return loc_loss + cls_loss


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


def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
    max_num_instances = output_shape[0]
    data = tf.reshape(data, [-1, *output_shape[1:]])
    num_instances = tf.shape(data)[0]
    assert_length = tf.Assert(
        tf.less_equal(num_instances, max_num_instances), [num_instances])
    with tf.control_dependencies([assert_length]):
        pad_length = max_num_instances - num_instances
    paddings = tf.fill([pad_length, *output_shape[1:]], tf.cast(pad_value, data.dtype))
    padded_data = tf.concat([data, paddings], axis=0)
    padded_data = tf.reshape(padded_data, output_shape)
    return padded_data


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


def draw_bboxes2(img, boxes, classes=None, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16, figsize=(10, 10)):
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