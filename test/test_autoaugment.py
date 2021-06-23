from PIL import Image

import numpy as np

import tensorflow as tf
from hanser.transform.autoaugment.imagenet import autoaugment, randaugment, rand_or_auto_augment

im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = tf.convert_to_tensor(np.array(im))
x2 = rand_or_auto_augment(x, 2, 10)
Image.fromarray(x2.numpy()).show()