import tensorflow as tf

from hanser.detection.iou import bbox_iou
from hanser.ops import index_put

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

    shape = tf.shape(ious)
    num_gts, num_bboxes = shape[0], shape[1]

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
        # Low-quality matching will overwrite the assigned_gt_inds assigned
        # in Step 3. Thus, the assigned gt might not be the best one for
        # prediction.
        # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        # bbox 1 will be assigned as the best target for bbox A in step 3.
        # However, if GT bbox 2's gt_argmax_ious = A, bbox A's
        # assigned_gt_inds will be overwritten to be bbox B.
        # This might be the reason that it is not used in ROI Heads.
        # for each gt, which anchor best ious with it
        # for each gt, the max iou of all proposals
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

    # if gt_labels is not None:
    #     pos_mask = assigned_gt_inds > 0
    #     assigned_labels = tf.gather(gt_labels, tf.where(pos_mask, assigned_gt_inds, 1) - 1)
    #     assigned_labels = tf.where(pos_mask, assigned_labels, 0)
    # else:
    #     assigned_labels = None

    # return num_gts, assigned_gt_inds, max_ious, assigned_labels
    return assigned_gt_inds
