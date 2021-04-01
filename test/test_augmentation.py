# import tensorflow as tf
#
# n = 4
# image = tf.random.normal((n, 160, 160, 3))
# label = tf.one_hot(tf.random.uniform((n,), 0, 1000, dtype=tf.int32), 1000, dtype=tf.float32)
# alpha = 1.0
# hard = False
#
# from hanser.transform import mixup_batch
# mixup_batch(image, label, 0.2)

from PIL import Image

import numpy as np

import tensorflow as tf
from hanser.transform import photo_metric_distortion

im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))
x2 = photo_metric_distortion(x)
Image.fromarray(x2.numpy()).show()