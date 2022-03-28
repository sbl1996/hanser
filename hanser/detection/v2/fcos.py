from toolz import curry
import numpy as np
import tensorflow as tf

from hanser.detection.v2.loss import iou_loss
from hanser.detection.v2.utils import mlvl_concat
from hanser.losses import focal_loss, reduce_loss
from hanser.ops import _pair, get_shape, to_float, all_reduce_mean
from hanser.detection.v2.anchor import grid_points


INF = 100000000
REGRESS_RANGES = ((0, 64), (64, 128), (128, 256), (256, 512), (512, INF))


class FCOSHead:

    def __init__(self, size, strides):
        strides = [_pair(s) for s in strides]
        featmap_sizes = [
            (size[0] // s[0], size[1] // s[1]) for s in strides]
        mlvl_points = grid_points(featmap_sizes, strides, center_offset=0.5)
        num_level_points = [p.shape[0] for p in mlvl_points]
        points = tf.concat(mlvl_points, axis=0)

        self.size = size
        self.strides = strides
        self.featmap_sizes = featmap_sizes
        self.num_level_points = num_level_points
        self.points = points

    def match(self, gt_bboxes, gt_labels, regress_ranges=REGRESS_RANGES, radius=1.5, centerness=True):
        """Compute regression, classification and centerness targets for points
        in multiple images.
        Args:
            gt_bboxes (tf.Tensor): (num_gts, 4)
            gt_labels (tf.Tensor): (num_gts,)
        Returns:
            bbox_targets (tf.Tensor): (num_points, 4)
            labels (tf.Tensor): (num_points,)
            centerness (tf.Tensor): (num_points,)
        """
        assert len(regress_ranges) == len(self.strides)
        points = self.points
        num_level_points = self.num_level_points
        strides = self.strides
        TINF = tf.constant(INF, dtype=tf.float32)

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
        regress_ranges = mlvl_concat(regress_ranges, num_level_points)

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

            radius = mlvl_concat(np.array(strides) * radius, num_level_points)
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
        areas = tf.where(cond, areas, TINF)
        min_area_inds = tf.argmin(areas, axis=1, output_type=tf.int32)
        min_area = tf.gather(areas, min_area_inds, axis=1, batch_dims=1)

        pos = min_area != TINF
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

    @curry
    def loss(self, y_true, y_pred,
             box_loss_fn=iou_loss(mode='giou', offset=True),
             cls_loss_fn=focal_loss(alpha=0.25, gamma=2.0),
             box_loss_weight=2.0, quantity_weighted=True):

        bbox_targets = y_true['bbox_target']
        labels = y_true['label']
        centerness_t = y_true['centerness']

        bbox_preds = y_pred['bbox_pred']
        cls_scores = y_pred['cls_score']
        centerness = y_pred['centerness']

        pos_weight = to_float(labels != 0)
        total_pos = tf.reduce_sum(pos_weight)
        total_pos = all_reduce_mean(total_pos)

        loss_cls = cls_loss_fn(labels, cls_scores, reduction='sum') / total_pos

        loss_centerness = tf.nn.sigmoid_cross_entropy_with_logits(centerness_t, centerness)
        loss_centerness = reduce_loss(loss_centerness, pos_weight, reduction='sum') / total_pos

        box_losses_weight = pos_weight
        box_loss_avg_factor = total_pos
        if quantity_weighted:
            box_losses_weight = centerness_t
            box_loss_avg_factor = tf.reduce_sum(centerness_t)
            box_loss_avg_factor = all_reduce_mean(box_loss_avg_factor)

        loss_box = box_loss_fn(bbox_targets, bbox_preds, weight=box_losses_weight, reduction='sum') / box_loss_avg_factor

        loss = loss_cls + loss_centerness + loss_box * box_loss_weight
        return loss