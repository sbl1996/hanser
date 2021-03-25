import numpy as np

import tensorflow as tf
from hanser.models.imagenet.resnet_vd import resnet50

model = resnet50(zero_init_residual=False)
model.build((None, 224, 224, 3))
model.summary()

ckpt = tf.train.Checkpoint(model=model)
ckpt.read()
# np.save(model.get_weights()