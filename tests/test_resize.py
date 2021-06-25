import numpy as np
from PIL import Image

import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

x = np.array([[0, 1, 2]], dtype=np.float64)
im =  Image.fromarray(x)
im2 = im.resize((4, 1), resample=Image.BILINEAR)
x2 = np.array(im2)
t = tf.cast(x, tf.float32)[:, :, None]
t2 = tf.image.resize(t, (1, 4))
tf.compat.v1.image.resize(t, (1, 4), method=tf.compat.v1.image.ResizeMethod.BILINEAR, align_corners=False)

tt = torch.from_numpy(x).to(torch.float32)
F.interpolate(tt[None, None], (1, 4), mode='bilinear', align_corners=False)

# Conclusion:
# tf.image.resize == Image.resize == F.interpolate(align_corners=False)
# Do not use tf.compat.v1.image.resize, regradless of align_corners