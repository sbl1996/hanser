import tensorflow as tf
from hanser.tpu import setup

from hanser.metrics import mean_iou

setup([], fp16=True)
strategy = tf.distribute.get_strategy()

@tf.function
def calculate_miou(y_true, y_pred):
    return mean_iou(y_true, y_pred, 4, ignore_index=255)

y_true = tf.random.uniform((4, 20, 20), minval=0, maxval=4, dtype=tf.int32)
y_pred = tf.random.uniform((4, 20, 20), minval=0, maxval=4, dtype=tf.int32)

strategy.run(calculate_miou, (y_true, y_pred))