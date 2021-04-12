import tensorflow as tf

from hanser.detection.iou import bbox_iou
from hanser.ops import index_put, get_shape, l2_norm


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
        if gt_max_assign_all:
            pass
        else:
            assigned_gt_inds = index_put(
                assigned_gt_inds, gt_argmax_ious,
                tf.where(
                    gt_max_ious > min_pos_iou,
                    tf.range(1, num_gts + 1),
                    tf.gather(assigned_gt_inds, gt_argmax_ious)
                )
            )

    return assigned_gt_inds



def atss_assign(bboxes, num_level_bboxes, gt_bboxes, topk=9):

    NINF = tf.constant(-100000000, dtype=tf.float32)
    num_gts = get_shape(gt_bboxes, 0)
    num_bboxes = get_shape(bboxes, 0)

    if num_gts == 0:
        # No truth, assign everything to background
        return tf.fill((num_bboxes,), tf.constant(0, dtype=tf.int32))

    # compute iou between all bbox and gt
    # (num_gts, num_bboxes)
    ious = bbox_iou(gt_bboxes, bboxes)

    # compute center distance between all bbox and gt
    gt_points = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
    bboxes_points = (bboxes[:, :2] + bboxes[:, 2:]) / 2.0
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
