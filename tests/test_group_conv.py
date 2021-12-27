import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, Activation

def setup_tpu():
    tf.keras.backend.clear_session()

    tpu_address = get_colab_tpu_address()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.TPUStrategy(tpu)
    tf.distribute.experimental_set_strategy(strategy)

def get_colab_tpu_address():
    tpu_address = os.environ.get("COLAB_TPU_ADDR")
    if tpu_address is not None:
        tpu_address = "grpc://" + tpu_address
    return tpu_address

setup_tpu()
strategy = tf.distribute.get_strategy()

def local_results(values):
    def func(x):
        if "PerReplica" in type(x).__name__:
            x = strategy.experimental_local_results(x)
            return tf.concat(x, axis=0)
        return x
    return tf.nest.map_structure(func, values)


def run_tf(fn, *args):
    if strategy is None:
        return fn(*args)
    else:
        return local_results(strategy.run(fn, args))

class NaiveGroupConv2D(Layer):

    def __init__(self, out_channels, kernel_size, groups, stride=1, padding='same'):
        super().__init__()
        self.groups = groups
        D_out = out_channels // groups
        self.convs = [
            Conv2D(D_out, kernel_size=kernel_size, strides=stride, padding=padding)
            for _ in range(groups)
        ]

    def call(self, x):
        xs = tf.split(x, self.groups, axis=-1)
        xs = [
            conv(x) for conv, x in zip(self.convs, xs)
        ]
        x = tf.concat(xs, axis=-1)
        return x

n, h, w, c = 32, 7, 7, 4

model1 = tf.keras.Sequential([
    Conv2D(8, kernel_size=3, padding='same', groups=2),
])
model1.build((None, h, w, c))

model2 = tf.keras.Sequential([
    NaiveGroupConv2D(8, kernel_size=3, padding='same', groups=2),
])
model2.build((None, h, w, c))



@tf.function
def test_train_tf(model, x):
    with tf.GradientTape() as tape:
        y = model(x, training=True)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, model.trainable_variables)
    return y, loss, grads

X = tf.random.normal((n, h, w, c))
ds = tf.data.Dataset.from_tensor_slices((X,)).batch(batch_size=16, drop_remainder=True)
ds_t = strategy.experimental_distribute_dataset(ds)
x = next(iter(ds_t))
y, loss, grads = run_tf(test_train_tf)
