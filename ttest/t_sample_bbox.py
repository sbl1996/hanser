import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from hanser.transform import sample_distorted_bounding_box

def sample_tf(shape, scale, ratio, min_object_covered=0.1):
    bbox = tf.zeros((1, 0, 4), dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        aspect_ratio_range=ratio,
        area_range=scale,
        min_object_covered=min_object_covered,
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
# res3 = []
# res4 = []
for i in range(n):
    # res1.append(sample_tf(shape, (0.05, 1.0), ratio, min_object_covered=0).numpy())
    # res2.append(sample_tf(shape, (0.08, 1.0), ratio, min_object_covered=0).numpy())
    # res2.append(tf.stack(sample_distorted_bounding_box(shape, scale, ratio)).numpy())
    # res3.append(tf.stack(sample_distorted_bounding_box(shape, scale, ratio, sample_log_ratio=False)).numpy())
    # res4.append(sample_tf(shape, scale, ratio, min_object_covered=0).numpy())
    res1.append(sample_tf(shape, (0.01, 1.0), ratio, min_object_covered=0.1).numpy())
    res2.append(sample_tf(shape, (0.1, 1.0), ratio, min_object_covered=0.0).numpy())

res1 = np.stack(res1, axis=0)
res2 = np.stack(res2, axis=0)
# res3 = np.stack(res3, axis=0)
# res4 = np.stack(res4, axis=0)、

j = 0
plt.hist(res1[:, j]); plt.hist(res2[:, j], alpha=0.5)
plt.show()