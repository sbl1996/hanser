import tensorflow as tf
from hanser.tpu import setup

from hanser.ops import gumbel_softmax

setup([], fp16=True)
strategy = tf.distribute.get_strategy()
tf.distribute.experimental_set_strategy(strategy)
weights = tf.Variable(tf.random.normal([3]), trainable=True)

@tf.function
def step_fn(weights):
    ret, index = gumbel_softmax(weights, 1.0, True, return_index=True)
    return ret, index

strategy.run(step_fn, (weights,))