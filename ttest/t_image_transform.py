import numpy as np
import tensorflow as tf

from hanser.transform import brightness, contrast, color


from PIL import Image
im = Image.open('/Users/hrvvi/Downloads/images/cat1.jpeg')
im = im.crop((66, 0, 234, 168)).resize((224, 224))

x = np.array(im)

y1 = contrast(x, 0.3)
y2 = tf.cast(contrast(tf.cast(x, tf.float32) / 255.0, 0.3) * 255, tf.uint8)
Image.fromarray(y1.numpy().astype(np.uint8)).show()

im2 = Image.open('/Users/hrvvi/Downloads/images/cat2.jpeg')
im2 = im2.crop((0, 0, 224, 224)).resize((224, 224))
