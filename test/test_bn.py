import tensorflow as tf
from hanser.models.layers import Norm, set_default

set_default(['bn', 'affine'], False)
x = tf.random.normal([1, 2, 2, 3])
bn = Norm(3)
y1 = bn(x, training=False)
y2 = bn(x, training=True)