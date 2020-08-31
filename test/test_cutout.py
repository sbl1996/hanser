from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from hanser.transform import cutout

im = Image.open("/Users/hrvvi/Downloads/cat.jpeg")
im = im.crop([66, 0, 66 + 168, 168]).resize((224, 224))
x = np.array(im)
t = tf.convert_to_tensor(x, dtype=tf.float32) / 255

# t1 = cutout(t, 112)
t1 = tfa.image.random_cutout(t[None], 112)[0]
x1 = (t1 * 255).numpy().astype(np.uint8)
Image.fromarray(x1).show()
# Image.fromarray(x - x1).show()