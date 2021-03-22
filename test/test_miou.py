import tensorflow as tf

from tensorflow.keras.metrics import MeanIoU as MeanIoU2

from hanser.tpu import setup
from hanser.train.metrics import MeanIoU

setup([], fp16=True)
strategy = tf.distribute.get_strategy()

num_classes = 8
miou = MeanIoU(num_classes)
miou2 = MeanIoU2(num_classes, dtype=tf.float32)

@tf.function
def train_step(y_true, y_pred):
    miou.update_state(y_true, y_pred)
    miou2.update_state(y_true, y_pred)

@tf.function
def train_batch(y_true, y_pred):
    train_step(y_true, y_pred)
    # strategy.run(train_step, (y_true, y_pred))

@tf.function
def run_steps(step_fn, n_steps):
    for i in tf.range(n_steps):
        y_pred = tf.random.uniform((8, 1024, 2048), minval=0, maxval=num_classes, dtype=tf.int32)
        y_true = tf.random.uniform((8, 1024, 2048), minval=0, maxval=num_classes, dtype=tf.int32)

        step_fn(y_true, y_pred)
    # return miou.result()
    return miou.result(), miou2.result()

run_steps(train_batch, 4)