import tensorflow as tf
from hanser.models.layers import Conv2d
from hanser.models.modules import DropBlock
from hanser.distribute import setup_tpu

setup_tpu(fp16=False)
strategy = tf.distribute.get_strategy()


def run_tf(fn, *args):
    if strategy is None:
        return fn(*args)
    else:
        return strategy.run(fn, args)


model = tf.keras.Sequential([
    Conv2d(2, 3, kernel_size=3, padding='same', norm='def', act='def'),
    DropBlock(0.9, block_size=7),
])
model.build((None, 28, 28, 2))


@tf.function
def test_train_tf(x):
    with tf.GradientTape() as tape:
        x = tf.pad(x, paddings=[(0, 0), (1, 1,), (1, 1), (0, 0)], mode='REFLECT'),
        y = model(x, training=True)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, model.trainable_variables)
    return y, loss, grads

x_tf = tf.random.normal((2, 28, 28, 2))
y, loss, grads = run_tf(test_train_tf, x_tf)
