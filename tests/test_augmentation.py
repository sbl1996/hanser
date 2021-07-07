import math

from PIL import Image

import numpy as np

import tensorflow as tf
from hanser.transform import random_erasing, IMAGENET_MEAN, IMAGENET_STD, cutout, cutout2, cutout3
from hanser.transform.autoaugment.cifar import autoaugment
im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))
xc = x[:, 66:234:]
# x = tf.cast(x, dtype=tf.float32)
# x = (x - IMAGENET_MEAN) / IMAGENET_STD
# x2 = cutout2(x, 112, 'uniform')
x2 = autoaugment(x, "CIFAR10")
# x2 = shear_x(xt, 1, 0)
# x2 = x2 * IMAGENET_STD + IMAGENET_MEAN
x2 = x2.numpy()
# x2 = x2.astype(np.uint8)
Image.fromarray(x2).show()