import random
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf

from hanser.transform.autoaugment.cifar import *

img = Image.open("/Users/hrvvi/Downloads/cat.jpeg")
im = np.array(img)

level = 8

magnitude = np.linspace(0.0, 0.9, 10)[level]
im1 = np.array(ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])))

x = tf.convert_to_tensor(im)
# im2 = color(x, 1.1).numpy()
im2 = NAME_TO_FUNC["color"](x, level).numpy()
np.all(im1 == im2)
# im3 = shear_y(x, 0.5, 0)
Image.fromarray(im1).show()