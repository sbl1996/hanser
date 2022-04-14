import numpy as np
import tensorflow as tf

from hanser.detection.iou import bbox_iou
from hanser.detection.bbox import centerness_target
from hanser.ops import index_put, get_shape, l2_norm, _pair, _meshgrid


def max_iou_assign(bboxes, gt_bboxes, pos_iou_thr, neg_iou_thr,
                   min_pos_iou=.0, match_low_quality=True, gt_max_assign_all=False):
    """Assign a corresponding gt bbox or background to each bbox.

    This method assign a gt bbox to every bbox (proposal/anchor), each bbox
    will be assigned with -1, or a semi-positive number. -1 means negative
    sample, semi-positive number is the index (0-based) of assigned gt.
    The assignment is done in following steps, the order matters.

    1. assign every bbox to the background
    2. assign proposals whose iou with all gts < neg_iou_thr to 0
    3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
       assign it to that bbox
    4. for each gt bbox, assign its nearest proposals (may be more than
       one) to itself

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
        gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """
    assert not gt_max_assign_all, "Not implemented."
    num_gts = get_shape(gt_bboxes, 0)
    num_bboxes = get_shape(bboxes, 0)

    if num_gts == 0:
        # No truth, assign everything to background
        return tf.fill((num_bboxes,), tf.constant(0, dtype=tf.int32))

    ious = bbox_iou(gt_bboxes, bboxes)

    # 1. assign -1 by default
    assigned_gt_inds = tf.fill((num_bboxes,), tf.constant(-1, dtype=tf.int32))

    # for each anchor, which gt best ious with it
    # for each anchor, the max iou of all gts
    max_ious = tf.reduce_max(ious, axis=0)
    argmax_ious = tf.argmax(ious, axis=0, output_type=tf.int32)

    # 2. assign negative: below
    # the negative inds are set to be 0
    mask = max_ious < neg_iou_thr
    assigned_gt_inds = tf.where(mask, 0, assigned_gt_inds)

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_ious >= pos_iou_thr
    assigned_gt_inds = tf.where(pos_inds, argmax_ious + 1, assigned_gt_inds)

    if match_low_quality:
        gt_max_ious = tf.reduce_max(ious, axis=1)
        gt_argmax_ious = tf.argmax(ious, axis=1, output_type=tf.int32)
        assigned_gt_inds = index_put(
            assigned_gt_inds, gt_argmax_ious,
            tf.where(
                gt_max_ious > min_pos_iou,
                tf.range(1, num_gts + 1),
                tf.gather(assigned_gt_inds, gt_argmax_ious)
            )
        )

    return assigned_gt_inds


def atss_assign(bboxes, num_level_bboxes, gt_bboxes, topk=9,
                is_points=False, strides=(8, 16, 32, 64, 128), scale=8):

    NINF = tf.constant(-100000000, dtype=tf.float32)
    num_gts = get_shape(gt_bboxes, 0)
    num_bboxes = get_shape(bboxes, 0)

    if num_gts == 0:
        # No truth, assign everything to background
        return tf.fill((num_bboxes,), tf.constant(0, dtype=tf.int32))

    gt_points = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
    if is_points:
        points = bboxes
        bbox_lengths = tf.constant([
            x * scale for s, n in zip(strides, num_level_bboxes) for x in [s] * n], dtype=points.dtype) / 2
        bboxes = tf.concat([points - bbox_lengths, points + bbox_lengths], axis=1)
        bboxes_points = points
    else:
        # compute center distance between all bbox and gt
        bboxes_points = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0

    # compute iou between all bbox and gt
    # (num_gts, num_bboxes)
    ious = bbox_iou(gt_bboxes, bboxes)

    # (num_gts, num_bboxes) for topk
    distances = l2_norm(bboxes_points[None, :, :] - gt_points[:, None, :], sqrt=True)

    # Selecting candidates based on the center distance
    candidate_idxs = []
    start_idx = 0
    for level, bboxes_per_level in enumerate(num_level_bboxes):
        # on each pyramid level, for each gt,
        # select k bbox whose center are closest to the gt center
        end_idx = start_idx + bboxes_per_level
        distances_per_level = distances[:, start_idx:end_idx]
        selectable_k = min(topk, bboxes_per_level)
        topk_idxs_per_level = tf.math.top_k(-distances_per_level, selectable_k, sorted=False)[1]
        candidate_idxs.append(topk_idxs_per_level + start_idx)
        start_idx = end_idx

    # (num_gts, num_levels * topk)
    candidate_idxs = tf.concat(candidate_idxs, axis=1)

    # get corresponding iou for the these candidates, and compute the
    # mean and std, set mean + std as the iou threshold
    # (num_gts, num_levels * topk)
    candidate_ious = tf.gather(ious, candidate_idxs, axis=1, batch_dims=1)
    ious_mean_per_gt = tf.reduce_mean(candidate_ious, axis=1)
    ious_std_per_gt = tf.math.reduce_std(candidate_ious, axis=1)
    ious_thr_per_gt = ious_mean_per_gt + ious_std_per_gt
    # tf.print(ious_thr_per_gt, summarize=-1)
    is_pos = candidate_ious >= ious_thr_per_gt[:, None]

    # limit the positive sample's center in gt
    # (num_gts, num_levels * topk, 2)
    ep_bboxes_cycx = tf.gather(bboxes_points, candidate_idxs, axis=0, batch_dims=1)
    # calculate the left, top, right, bottom distance between positive
    # bbox center and gt side
    t_ = ep_bboxes_cycx[..., 0] - gt_bboxes[:, None, 0]
    l_ = ep_bboxes_cycx[..., 1] - gt_bboxes[:, None, 1]
    b_ = gt_bboxes[:, None, 2] - ep_bboxes_cycx[..., 0]
    r_ = gt_bboxes[:, None, 3] - ep_bboxes_cycx[..., 1]
    is_in_gts = tf.reduce_min([t_, l_, b_, r_], axis=0) > 0.01
    is_pos = is_pos & is_in_gts

    candidate_idxs = candidate_idxs + tf.range(num_gts)[:, None] * num_bboxes

    # if an anchor box is assigned to multiple gts,
    # the one with the highest IoU will be selected.
    ious_inf = tf.fill((num_gts * num_bboxes,), NINF)
    index = candidate_idxs[is_pos]
    # tf.print("final", tf.gather(tf.reshape(ious, (-1,)), index), summarize=-1)
    ious_inf = index_put(ious_inf, index, tf.gather(tf.reshape(ious, (-1,)), index))
    ious_inf = tf.transpose(tf.reshape(ious_inf, (num_gts, num_bboxes)))

    argmax_ious = tf.argmax(ious_inf, axis=1, output_type=tf.int32)
    max_ious = tf.gather(ious_inf, argmax_ious, axis=1, batch_dims=1)

    assigned_gt_inds = tf.where(max_ious != NINF, argmax_ious + 1, 0)

    # pos = assigned_gt_inds != 0
    # pos_ious = tf.gather(ious_inf[pos], assigned_gt_inds[pos] - 1, axis=1, batch_dims=1)
    return assigned_gt_inds



def yolo_assign(bboxes, gt_bboxes, num_level_bboxes,
                strides=(8, 16, 32), ignore_iou_thr=0.5):
    # This implementation is consistent with the original paper,
    # but different from either PaddleDetection or MMDetection.
    # Reference:
    # PaddleDetection
    #   https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/ppdet/data/transform/batch_operators.py#L176-L287
    #   https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/ppdet/modeling/losses/yolo_loss.py#L73-L87
    # MMDetection
    #   https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/grid_assigner.py#L10-L156
    #   https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L831-L866
    num_gts = get_shape(gt_bboxes, 0)
    num_bboxes = get_shape(bboxes, 0)

    if num_gts == 0:
        # No truth, assign everything to background
        return tf.fill((num_bboxes,), tf.constant(0, dtype=tf.int32))

    ious = bbox_iou(gt_bboxes, bboxes)

    # P1. assign -1 by default
    assigned_gt_inds = tf.fill((num_bboxes,), tf.constant(-1, dtype=tf.int32))

    # for each anchor, the max iou of all gts
    max_ious = tf.reduce_max(ious, axis=0)

    # P2. assign negative: below
    # the negative inds are set to be 0
    mask = max_ious < ignore_iou_thr
    assigned_gt_inds = tf.where(mask, 0, assigned_gt_inds)

    # P3. assign positive
    bboxes_centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
    gt_bboxes_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2

    radius = mlvl_concat(
        np.array(strides) * 0.5, num_level_bboxes)
    radius = radius[None, :]
    same_grid_mask = tf.reduce_max(tf.abs(
        bboxes_centers[None, :, :] - gt_bboxes_centers[:, None, :]), axis=-1) < radius
    ious = tf.where(same_grid_mask, ious, -1)

    gt_argmax_ious = tf.argmax(ious, axis=1, output_type=tf.int32)
    assigned_gt_inds = index_put(
        assigned_gt_inds, gt_argmax_ious, tf.range(1, num_gts + 1))

    # anchor
    # 1. overlap without any gt                 0
    # 2. overlap with some gt (is max, > 0.5)   X
    # 3. overlap with some gt (is max, < 0.5)   X
    # 4. overlap with some gt (not max, > 0.5)  -1
    # 5. overlap with some gt (not max, < 0.5)  0
    # we don't consider that the anchor is the max of multiple gts

    # P1. -1 -1 -1 -1 -1
    # P2. 0 -1  0 -1  0
    # P3. 0  X  X -1  0

    return assigned_gt_inds


def mlvl_concat(xs, reps, dtype=tf.float32):
    xs = np.array(xs)
    ndim = len(xs.shape)
    xs = np.concatenate([
        np.tile(xs[i][None], (n,) + (1,) * (ndim - 1))
        for i, n in enumerate(reps)
    ], axis=0)
    return tf.constant(xs, dtype)


def grid_points(featmap_sizes, strides, center_offset=0):
    assert len(featmap_sizes) == len(strides)
    strides = [_pair(s) for s in strides]
    mlvl_points = []
    for featmap_size, stride in zip(featmap_sizes, strides):
        feat_h, feat_w = featmap_size[0], featmap_size[1]
        point_y = tf.range(0, feat_h, dtype=tf.float32) * stride[0]
        point_x = tf.range(0, feat_w, dtype=tf.float32) * stride[1]
        if center_offset:
            point_y = point_y + stride[0] * center_offset
            point_x = point_x + stride[1] * center_offset
        point_yy, point_xx = _meshgrid(point_y, point_x, row_major=False)
        points = tf.stack([point_yy, point_xx], axis=-1)
        mlvl_points.append(points)
    return mlvl_points

INF = 100000000


def max_iou_match(gt_bboxes, gt_labels, bbox_coder, pos_iou_thr=0.5, neg_iou_thr=0.4,
                   min_pos_iou=0.0, match_low_quality=True, encode_bbox=True, centerness=False):
    anchors = bbox_coder.anchors
    assigned_gt_inds = max_iou_assign(
        anchors, gt_bboxes, pos_iou_thr, neg_iou_thr, min_pos_iou, match_low_quality)
    return encode_target(
        gt_bboxes, gt_labels, assigned_gt_inds,
        bbox_coder=bbox_coder, centerness=centerness, encode_bbox=encode_bbox)


def yolo_match(gt_bboxes, gt_labels, bbox_coder, num_level_bboxes,
               strides=(8, 16, 32), ignore_iou_thr=0.5, encode_bbox=True):
    anchors = bbox_coder.anchors
    assigned_gt_inds = yolo_assign(anchors, gt_bboxes, num_level_bboxes, strides, ignore_iou_thr)
    return encode_target(
        gt_bboxes, gt_labels, assigned_gt_inds,
        bbox_coder=bbox_coder, encode_bbox=encode_bbox)


def atss_match(gt_bboxes, gt_labels, anchors, num_level_bboxes, topk=9, centerness=True):
    assigned_gt_inds = atss_assign(anchors, num_level_bboxes, gt_bboxes, topk=topk)
    return encode_target(
        gt_bboxes, gt_labels, assigned_gt_inds, anchors=anchors,
        encode_bbox=False, centerness=centerness, with_ignore=False)


def fcos_match(gt_bboxes, gt_labels, points, num_level_points,
               strides=(8, 16, 32, 64, 128), radius=1.5,
               regress_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
               centerness=True):
    strides = [_pair(s) for s in strides]
    INF = tf.constant(100000000, dtype=tf.float32)

    num_points = get_shape(points, 0)
    num_gts = get_shape(gt_bboxes, 0)

    if num_gts == 0:
        bbox_targets = tf.zeros([num_points, 4], dtype=tf.float32)
        labels = tf.zeros([num_points, ], dtype=tf.int32)
        if not centerness:
            return bbox_targets, labels
        centerness_t = tf.zeros([num_points, ], dtype=tf.float32)
        return bbox_targets, labels, centerness_t

    # (num_points, 2)
    regress_ranges = mlvl_concat(
        regress_ranges, num_level_points)

    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1])

    # (num_points, num_gts, *)
    areas = tf.tile(areas[None], (num_points, 1))
    regress_ranges = tf.tile(
        regress_ranges[:, None, :], (1, num_gts, 1))
    gt_bboxes = tf.tile(gt_bboxes[None], (num_points, 1, 1))
    points = tf.tile(points[:, None, :], (1, num_gts, 1))
    ys, xs = points[..., 0], points[..., 1]

    t = ys - gt_bboxes[..., 0]
    l = xs - gt_bboxes[..., 1]
    b = gt_bboxes[..., 2] - ys
    r = gt_bboxes[..., 3] - xs
    bbox_targets = tf.stack((t, l, b, r), axis=-1)

    if radius is not None:
        # center sampling
        # condition1: inside a `center bbox`
        centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2

        radius = mlvl_concat(
            np.array(strides) * radius, num_level_points)
        radius = radius[:, None, :]
        mins = centers - radius
        maxs = centers + radius
        center_gts_tl = tf.maximum(mins, gt_bboxes[..., :2])
        center_gts_br = tf.minimum(maxs, gt_bboxes[..., 2:])

        center_gts_t, center_gts_l = center_gts_tl[..., 0], center_gts_tl[..., 1]
        center_gts_b, center_gts_r = center_gts_br[..., 0], center_gts_br[..., 1]
        inside_gt_bbox_mask = tf.math.reduce_all(
            [ys > center_gts_t,
             xs > center_gts_l,
             ys < center_gts_b,
             xs < center_gts_r], axis=0)
    else:
        # condition1: inside a gt bbox
        inside_gt_bbox_mask = tf.reduce_min(bbox_targets, axis=-1) > 0

    # condition2: limit the regression range for each location
    max_regress_distance = tf.reduce_max(bbox_targets, axis=-1)
    inside_regress_range = (
        (max_regress_distance >= regress_ranges[..., 0]) &
        (max_regress_distance <= regress_ranges[..., 1]))

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    cond = inside_gt_bbox_mask & inside_regress_range
    areas = tf.where(cond, areas, INF)
    min_area_inds = tf.argmin(areas, axis=1, output_type=tf.int32)
    min_area = tf.gather(areas, min_area_inds, axis=1, batch_dims=1)

    pos = min_area != INF
    bbox_targets = tf.gather(
        bbox_targets, min_area_inds, axis=1, batch_dims=1)
    labels = tf.where(
        pos, tf.gather(gt_labels, min_area_inds), 0)

    if not centerness:
        return bbox_targets, labels

    t, l, b, r = [bbox_targets[:, i] for i in range(4)]
    centerness = tf.sqrt(
        (tf.minimum(l, r) / tf.maximum(l, r)) *
        (tf.minimum(t, b) / tf.maximum(t, b)))
    centerness = tf.where(pos, centerness, 0)

    return bbox_targets, labels, centerness


def dw_match(gt_bboxes, gt_labels, points, num_level_points, strides=(8, 16, 32, 64, 128)):
    strides = [_pair(s) for s in strides]
    INF = tf.constant(100000000, dtype=tf.float32)

    num_points = get_shape(points, 0)
    num_gts = get_shape(gt_bboxes, 0)

    if num_gts == 0:
        bbox_targets = tf.zeros([num_points, 4], dtype=tf.float32)
        labels = tf.zeros([num_points, ], dtype=tf.int32)
        centerness_t = tf.zeros([num_points, ], dtype=tf.float32)
        return bbox_targets, labels, centerness_t

    # (num_points, num_gts, *)
    points = tf.tile(points[:, None, :], (1, num_gts, 1))
    gt_bboxes = tf.tile(gt_bboxes[None], (num_points, 1, 1))
    ys, xs = points[..., 0], points[..., 1]
    t = ys - gt_bboxes[..., 0]
    l = xs - gt_bboxes[..., 1]
    b = gt_bboxes[..., 2] - ys
    r = gt_bboxes[..., 3] - xs
    bbox_targets = tf.stack((t, l, b, r), axis=-1)
    inside_gt_bbox_mask = tf.reduce_min(bbox_targets, axis=-1) > 0

    radius = mlvl_concat(np.array(strides), num_level_points)
    gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
    distances = (points - gt_centers) / radius
    mean, sigma = 0, 1.11
    center_prior = tf.math.exp(
        -(distances - mean) ** 2 / (2 * (sigma ** 2)))
    center_prior = tf.math.reduce_prod(center_prior, axis=-1)

    center_prior = tf.where(inside_gt_bbox_mask, center_prior, 0)
    # labels = tf.where(
    #     pos, tf.gather(gt_labels, min_area_inds), 0)

    return bbox_targets, center_prior



def encode_target(gt_bboxes, gt_labels, assigned_gt_inds,
                  bbox_coder=None, encode_bbox=True,
                  anchors=None, centerness=False, with_ignore=True):
    if centerness:
        assert bbox_coder is not None or anchors is not None
    num_gts = get_shape(gt_bboxes, 0)
    num_anchors = get_shape(assigned_gt_inds, 0)
    bbox_targets = tf.zeros([num_anchors, 4], dtype=tf.float32)
    labels = tf.zeros([num_anchors,], dtype=tf.int32)
    if centerness:
        centerness_t = tf.zeros([num_anchors,], dtype=tf.float32)

    pos = assigned_gt_inds > 0
    indices = tf.range(num_anchors, dtype=tf.int32)[pos]

    if num_gts == 0 or get_shape(indices, 0) == 0:
        returns = [bbox_targets, labels]
        if centerness:
            returns.append(centerness_t)
        if with_ignore:
            ignore = tf.fill([num_anchors,], False)
            returns.append(ignore)
        return tuple(returns)

    ignore = assigned_gt_inds == -1
    assigned_gt_inds = tf.gather(assigned_gt_inds, indices) - 1
    assigned_gt_bboxes = tf.gather(gt_bboxes, assigned_gt_inds)
    assigned_gt_labels = tf.gather(gt_labels, assigned_gt_inds)

    if centerness:
        if bbox_coder is not None:
            anchors = bbox_coder.anchors
        assigned_anchors = tf.gather(anchors, indices, axis=0)
        assigned_centerness = centerness_target(assigned_gt_bboxes, assigned_anchors)
        centerness_t = index_put(centerness_t, indices, assigned_centerness)

    if encode_bbox:
        assigned_gt_bboxes = bbox_coder.encode(assigned_gt_bboxes, indices)

    ndims = assigned_gt_bboxes.shape[-1]
    if ndims != 4:
        bbox_targets = tf.zeros([num_anchors, ndims], dtype=tf.float32)
    bbox_targets = index_put(bbox_targets, indices, assigned_gt_bboxes)
    labels = index_put(labels, indices, assigned_gt_labels)

    returns = [bbox_targets, labels]
    if centerness:
        returns.append(centerness_t)
    if with_ignore:
        returns.append(ignore)
    return tuple(returns)