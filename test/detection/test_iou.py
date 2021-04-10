import tensorflow as tf

from hanser.detection import random_bboxes
from hanser.detection.iou import bbox_iou

b = random_bboxes((2, 5))
offset = tf.random.normal((2, 5), stddev=0.1).numpy()
b1 = b.numpy()
b1[..., 0] += offset
b1[..., 2] += offset
b1 = tf.convert_to_tensor(b1)
bbox_iou(b, b1, mode='iou', is_aligned=True)
bbox_iou(b, b1, mode='giou', is_aligned=True)
bbox_iou(b, b1, mode='diou', is_aligned=True)
bbox_iou(b, b1, mode='ciou', is_aligned=True)