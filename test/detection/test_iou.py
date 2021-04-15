import tensorflow as tf

from hanser.detection import random_bboxes
from hanser.detection.iou import bbox_iou, bbox_iou2, bbox_iou_offset

def bbox_offset(point, bbox):
    y, x = point[..., 0], point[..., 1]
    t = y - bbox[..., 0]
    l = x - bbox[..., 1]
    b = bbox[..., 2] - y
    r = bbox[..., 3] - x
    return tf.stack([t, l, b, r], axis=-1)

shape = (20, 50)

h, w = 300, 500
b = random_bboxes(shape) * [h, h, w, w]
c = (b[..., :2] + b[..., 2:]) / 2

# offset = tf.random.normal((2, 5), stddev=0.1).numpy()
# b1 = b.numpy()
# b1[..., 0] += offset
# b1[..., 2] += offset
# b1 = tf.convert_to_tensor(b1)
# b1 = tf.clip_by_value(b1, 0, 1)

# bbox_iou2(b, b1, mode='iou', is_aligned=True)
# bbox_iou2(b, b1, mode='giou', is_aligned=True)
# bbox_iou2(b, b1, mode='diou', is_aligned=True)
# bbox_iou2(b, b1, mode='ciou', is_aligned=True)


offset = tf.random.normal(shape + (4,), stddev=0.05)
b2 = tf.clip_by_value(b + offset, 0, 1)
c2 = (b2[..., :2] + b2[..., 2:]) / 2
cb = (c + c2) / 2 + tf.random.normal(shape + (2,), stddev=0.05)
cb = tf.clip_by_value(cb, 0, 1)

b2 = b2  * [h, h, w, w]
cb = cb * [h, w]

offset = bbox_offset(cb, b)
offset2 = bbox_offset(cb, b2)

for k in ['iou', 'giou', 'diou', 'ciou']:
    tf.debugging.assert_near(
        bbox_iou2(offset, offset2, mode=k, is_aligned=True, offset=True),
        bbox_iou_offset(offset, offset2, mode=k, is_aligned=True))

for k in ['iou', 'giou', 'diou', 'ciou']:
    tf.debugging.assert_near(
        bbox_iou2(b, b2, mode=k, is_aligned=True, offset=False),
        bbox_iou(b, b2, mode=k, is_aligned=True))


for k in ['iou', 'giou', 'diou', 'ciou']:
    tf.debugging.assert_near(
        bbox_iou2(b, b2, mode=k, is_aligned=False, offset=False),
        bbox_iou(b, b2, mode=k, is_aligned=False))

for k in ['iou', 'giou', 'diou', 'ciou']:
    tf.debugging.assert_near(
        bbox_iou2(offset, offset2, mode=k, is_aligned=False, offset=True),
        bbox_iou_offset(offset, offset2, mode=k, is_aligned=False))
