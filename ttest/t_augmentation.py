from PIL import Image

import numpy as np

import tensorflow as tf
from hanser.transform import IMAGENET_MEAN, IMAGENET_STD, translate_x
from hanser.transform.autoaugment.imagenet import randaugment


im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))
# xc = x[:, 66:234:]
# x = tf.cast(x, dtype=tf.float32)
# x = (x - IMAGENET_MEAN) / IMAGENET_STD
# x2 = translate_x(x, 96, 0)
x2 = randaugment(x, 1, 15, augmentation_space=['shearX'])
# xs = tf.stack([x, x, x], axis=0)
# xs2 = translate_x(xs, 96, 0)
# x2 = xs2[0]
# x2 = shear_x(xt, 1, 0)
# x2 = x2 * IMAGENET_STD + IMAGENET_MEAN
x2 = x2.numpy()
x2 = x2.astype(np.uint8)
Image.fromarray(x2).show()