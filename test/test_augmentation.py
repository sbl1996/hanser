import tensorflow as tf

n = 4
image = tf.random.normal((n, 160, 160, 3))
label = tf.one_hot(tf.random.uniform((n,), 0, 1000, dtype=tf.int32), 1000, dtype=tf.float32)
alpha = 1.0
hard = False

from hanser.transform import mixup_batch
mixup_batch(image, label, 0.2)