from toolz import curry

import tensorflow as tf

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


class BBoxCoder:

    def __init__(self, anchors, bbox_std=(1., 1., 1., 1.)):
        self.anchors = tf.constant(anchors, tf.float32)
        self.bbox_std = tf.constant(bbox_std, tf.float32)

    def encode(self, bboxes, idx=None):
        anchors = self.anchors
        if idx is not None:
            anchors = tf.gather(anchors, idx, axis=0)
        return bbox_encode(bboxes, anchors, self.bbox_std)

    def decode(self, bboxes, idx=None):
        anchors = self.anchors
        if idx is not None:
            anchors = tf.gather(anchors, idx, axis=0)
        return bbox_decode(bboxes, anchors, self.bbox_std)


class FCOSBBoxCoder:

    def __init__(self, points):
        self.points = tf.constant(points, tf.float32)

    def encode(self, bboxes, idx=None):
        points = self.points
        if idx is not None:
            points = tf.gather(points, idx, axis=0)
        # preds: (batch_size, num_points, 4)
        ys, xs = points[..., 0], points[..., 1]
        t_ = ys - bboxes[..., 0]
        l_ = xs - bboxes[..., 1]
        b_ = bboxes[..., 2] - ys
        r_ = bboxes[..., 3] - xs
        return tf.stack([t_, l_, b_, r_], axis=-1)

    def decode(self, preds, idx=None):
        points = self.points
        if idx is not None:
            points = tf.gather(points, idx, axis=0)
        # preds: (batch_size, num_points, 4)
        ys, xs = points[..., 0], points[..., 1]
        t = ys - preds[..., 0]
        l = xs - preds[..., 1]
        b = preds[..., 2] + ys
        r = preds[..., 3] + xs
        return tf.stack([t, l, b, r], axis=-1)


class YOLOBBoxCoder:

    def __init__(self, anchors, strides, num_base_anchors, eps=1e-6):
        super(YOLOBBoxCoder, self).__init__()
        self.anchors = tf.constant(anchors, tf.float32)
        scales = [tf.repeat(s, n) for s, n in zip(strides, num_base_anchors)]
        self.scales = tf.cast(tf.concat(scales, axis=0), tf.float32)
        self.eps = eps

    def encode(self, bboxes, idx=None):
        anchors = self.anchors
        if idx is not None:
            anchors = tf.gather(anchors, idx, axis=0)
        boxes_yx = (bboxes[..., :2] + bboxes[..., 2:]) / 2
        boxes_hw = bboxes[..., 2:] - bboxes[..., :2]
        anchors_yx = (anchors[:, :2] + anchors[:, 2:]) / 2
        anchors_hw = anchors[:, 2:] - anchors[:, :2]
        thtw = tf.math.log(
            tf.maximum(boxes_hw / anchors_hw, self.eps))
        tytx = (boxes_yx - anchors_yx) / self.scales + 0.5
        tytx = tf.clip_by_value(tytx, self.eps, 1 - self.eps)
        target = tf.concat([tytx, thtw], axis=-1)
        return target

    def decode(self, pred, idx=None):
        anchors = self.anchors
        if idx is not None:
            anchors = tf.gather(anchors, idx, axis=0)
        anchors = tf.convert_to_tensor(anchors, pred.dtype)

        anchors_yx = (anchors[..., :2] + anchors[..., 2:]) / 2
        anchors_hw = anchors[..., 2:] - anchors[..., :2]

        bbox_yx = (pred[..., :2] - 0.5) * self.scales + anchors_yx
        bbox_hw = tf.exp(pred[..., 2:]) * anchors_hw

        bbox_hw_half = bbox_hw / 2
        bboxes = tf.concat([bbox_yx - bbox_hw_half, bbox_yx + bbox_hw_half], axis=-1)
        return bboxes


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
    centerness = tf.where(tf.math.is_nan(centerness), 0.0, centerness)
    return centerness
