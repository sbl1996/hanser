import numpy as np
import tensorflow as tf

from hanser.transform import mixup_batch, mixup_in_batch, cutmix_batch, cutmix_in_batch

n = 16
image = tf.random.normal((n, 32, 32, 3))
label = tf.one_hot(tf.random.uniform((n,), 0, 3, dtype=tf.int32), 3)
xt, yt = mixup_batch(image, label, 0.2, uniform=False)

xt2, yt2 = mixup_in_batch(image, label, 1.0, uniform=True)

xt3, yt3 = cutmix_batch(image, label, 1.0, uniform=True)
xt3, yt3 = cutmix_batch(image, label, 0.2, uniform=False)

xt4, yt4 = cutmix_in_batch(image, label, 1.0, uniform=True)
xt4, yt4 = cutmix_in_batch(image, label, 0.2, uniform=False)

from PIL import Image
im = Image.open('/Users/hrvvi/Downloads/cat.jpeg')
im = im.crop((66, 0, 234, 168)).resize((224,224))
im2 = Image.open('/Users/hrvvi/Downloads/cat2.jpg')
im2 = im2.crop((693, 0, 3413, 2720)).resize((224,224))

image = tf.convert_to_tensor([np.array(im), np.array(im2)], dtype=np.float32)
label = tf.one_hot([0, 2], 3)

xt, yt = cutmix_in_batch(image, label, 1.0, uniform=True)
print(yt)
xt = xt.numpy()
Image.fromarray(xt[0].astype(np.uint8)).show()