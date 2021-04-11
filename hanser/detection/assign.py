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
    ious = bbox_iou(gt_bboxes, bboxes)

    num_gts = get_shape(ious, 0)
    num_bboxes = get_shape(bboxes, 0)

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
    INF = 100000000
    num_gts = get_shape(ious, 0)
    num_bboxes = get_shape(bboxes, 0)

    # 1. assign 0 by default
    assigned_gt_inds = tf.fill((num_bboxes,), tf.constant(0, dtype=tf.int32))

    # compute iou between all bbox and gt
    ious = bbox_iou(bboxes, gt_bboxes)

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
    candidate_idxs = tf.concat(candidate_idxs, axis=0)

    # get corresponding iou for the these candidates, and compute the
    # mean and std, set mean + std as the iou threshold
    candidate_ious = tf.gather(ious, candidate_idxs, axis=0)
    ious_mean_per_gt = tf.reduce_mean(candidate_ious, axis=0)
    ious_std_per_gt = tf.math.reduce_std(candidate_ious, axis=0)
    ious_thr_per_gt = ious_mean_per_gt + ious_std_per_gt

    pos = candidate_ious >= ious_thr_per_gt[None, :]

    # limit the positive sample's center in gt
    for gt_idx in range(num_gts):
        candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
    ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        num_gt, num_bboxes).contiguous().view(-1)
    ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
        num_gt, num_bboxes).contiguous().view(-1)
    candidate_idxs = candidate_idxs.view(-1)

    # calculate the left, top, right, bottom distance between positive
    # bbox center and gt side
    l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
    t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
    r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
    b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
    is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
    is_pos = is_pos & is_in_gts

    # if an anchor box is assigned to multiple gts,
    # the one with the highest IoU will be selected.
    overlaps_inf = torch.full_like(overlaps,
                                   -INF).t().contiguous().view(-1)
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
    overlaps_inf = overlaps_inf.view(num_gt, -1).t()

    max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
    assigned_gt_inds[
        max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

    return AssignResult(
        num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
