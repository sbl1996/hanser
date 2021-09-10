from PIL import Image

import numpy as np
import tensorflow as tf

def rgb_to_grayscale(images):
    images = tf.convert_to_tensor(images)
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_image = tf.image.convert_image_dtype(images, tf.float32)

    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    rgb_weights = [0.299, 0.5870, 0.1140]
    gray_float = tf.tensordot(flt_image, rgb_weights, [-1, -1])
    gray_float = tf.expand_dims(gray_float, -1)
    return tf.image.convert_image_dtype(gray_float, orig_dtype)


im = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
x = np.array(im)
xl = np.array(im.convert("L"))
# im = im.crop([66, 0, 66 + 168, 168]).resize((224, 224))

tl = rgb_to_grayscale(tf.convert_to_tensor(x))

d = np.abs(tf.cast(tl, tf.int32)[:, :, 0].numpy() - xl.astype(np.int32))
np.bincount(d.flat)
