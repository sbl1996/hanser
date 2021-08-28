from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from hanser.models.modules import DropBlock


im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
im = im.crop((66, 0, 234, 168)).resize((20, 20))
x = tf.convert_to_tensor(np.array(im))
x = tf.cast(x, dtype=tf.float32)
# x2 = cutout2(x, 112, 'uniform')
xs = tf.stack([x for _ in range(4)], axis=0)

dp = DropBlock(0.9, block_size=3)
xs2 = dp(xs, training=True)
x2 = xs2[0]
# x2 = shear_x(xt, 1, 0)
x2 = x2.numpy()
x2 = x2.astype(np.uint8)
Image.fromarray(x2).show()
