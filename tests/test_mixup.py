import numpy as np
import tensorflow as tf

from hanser.transform import mixup_batch, mixup_in_batch, cutmix_batch, resizemix_batch, image_dimensions
from hhutil.io import fmt_path

def resizemix(data1, data2, alpha, beta):
    image1, label1 = data1
    image2, label2 = data2

    h, w, c = image_dimensions(image1, 3)

    tau = tf.random.uniform((), minval=alpha, maxval=beta)

    cut_w = tf.cast(tf.cast(w, tau.dtype) * tau, tf.int32)
    cut_h = tf.cast(tf.cast(h, tau.dtype) * tau, tf.int32)

    cx = tf.random.uniform((), cut_w - cut_w // 2, w - cut_w // 2 + 1, dtype=tf.int32)
    cy = tf.random.uniform((), cut_h - cut_h // 2, h - cut_h // 2 + 1, dtype=tf.int32)

    l = cx - (cut_w - cut_w // 2)
    t = cy - (cut_h - cut_h // 2)
    r = cx + cut_w // 2
    b = cy + cut_h // 2

    hi = tf.range(h)[:, None, None]
    mh = (hi >= t) & (hi < b)
    wi = tf.range(w)[None, :, None]
    mw = (wi >= l) & (wi < r)
    masks = tf.cast(tf.logical_not(mh & mw), image1.dtype)

    paste = tf.image.resize(image2, [cut_h, cut_w])
    paste = tf.pad(paste, [[t, h - b], [l, w - r], [0, 0]])
    image = image1 * masks + paste

    lam = tf.cast(tau ** 2, label1.dtype)
    label = label1 * lam + label2 * (1. - lam)
    return image, label



from PIL import Image
im = Image.open(fmt_path('~/Downloads/images/cat1.jpeg'))
im = im.crop((66, 0, 234, 168)).resize((224, 224))
im2 = Image.open(fmt_path('~/Downloads/images/cat2.jpeg'))
im2 = im2.crop((0, 0, 224, 224)).resize((224, 224))

image = tf.convert_to_tensor([np.array(im), np.array(im2)], dtype=np.float32)
label = tf.one_hot([0, 2], 3)

xt, yt = resizemix((image[0], label[0]), (image[1], label[1]), 0.1, 0.8)
xt = xt.numpy()
Image.fromarray(xt.astype(np.uint8)).show()


lams = []
for i in range(1):
    xt, yt = resizemix_batch(image, label, hard=True)
    lams.append(yt[0, 0].numpy())
print(yt)
xt = xt.numpy()
Image.fromarray(xt[0].astype(np.uint8)).show()

yp = tf.random.normal((2, 3))

from hanser.losses import CrossEntropy
criterion = CrossEntropy(label_smoothing=0.1)
loss1 = criterion(yt, yp)
lam = yt[0, 0]
yt_a = tf.one_hot([0, 2], 3)
yt_b = tf.one_hot([2, 0], 3)
loss2 = lam * criterion(yt_a, yp) + (1 - lam) * criterion(yt_b, yp)
np.testing.assert_allclose(loss1.numpy(), loss2.numpy())