import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from hanser.transform import sample_distorted_bounding_box

def sample_tf(shape, scale, ratio):
    bbox = tf.zeros((1, 0, 4), dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        aspect_ratio_range=ratio,
        area_range=scale,
        use_image_if_no_bounding_boxes=True)

    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    return tf.stack([offset_y, offset_x, target_height, target_width])


shape = (387, 469, 3)
scale = (0.05, 1.0)
ratio = (3/4, 4/3)

n = 10000

res1 = []
res2 = []
for i in range(n):
    res1.append(sample_distorted_bounding_box(shape, scale, ratio).numpy())
    res2.append(sample_tf(shape, scale, ratio).numpy())

res1 = np.stack(res1, axis=0)
res2 = np.stack(res2, axis=0)

j = 0
plt.hist(res1[:, j]); plt.hist(res2[:, j])