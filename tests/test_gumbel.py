import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from toolz import curry
from hanser.datasets.mnist import make_mnist_dataset

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.layers import Flatten
from tensorflow_addons.optimizers import AdamW
from hanser.transform import to_tensor, normalize, pad
from hanser.distribute import setup_runtime, distribute_datasets

from hanser.models.layers import Conv2d, Linear, Identity
from hanser.ops import gumbel_softmax

@curry
def transform(image, label, training):
    image = pad(image, 2)
    image, label = to_tensor(image, label)
    image = normalize(image, [0.1307], [0.3081])

    label = tf.one_hot(label, 10)

    return image, label


batch_size = 128
eval_batch_size = 256
ds_train, ds_test, steps_per_epoch, test_steps = \
    make_mnist_dataset(batch_size, eval_batch_size, transform, sub_ratio=0.01)

setup_runtime()
ds_train, ds_test = distribute_datasets(ds_train, ds_test)

class MixOp(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.ops = [
            # Identity(),
            Conv2d(4, 4, kernel_size=1, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=3, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=3, groups=4, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=5, norm='def', act='def'),
            Conv2d(4, 4, kernel_size=5, groups=4, norm='def', act='def'),
        ]

        self.weight = self.add_weight(
            name='weight', shape=(len(self.ops),), dtype=self.dtype,
            initializer=tf.keras.initializers.RandomNormal(stddev=1e-2), trainable=True,
        )

    # def _create_branch_fn(self, x, i, hardwts):
    #     return lambda: self.ops[i](x) * hardwts[i]

    def build(self, input_shape):
        for op in self.ops:
            op.build(input_shape)

    def call(self, x):
        hardwts, index = gumbel_softmax(self.alpha, tau=1.0, hard=True, return_index=True)
        # branch_fns = [
        #     self._create_branch_fn(x, i, hardwts)
        #     for i in range(3)
        # ]
        # return tf.switch_case(index, branch_fns)
        xs = [
            tf.cond(hardwts[i] == 1, lambda: self.ops[i](x) * hardwts[i], lambda: hardwts[i] * tf.zeros_like(x))
            for i in range(len(self.ops))
        ]
        return sum(xs)


class ConvNet(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stem = Conv2d(1, 4, kernel_size=1)
        self.op = MixOp()

        self.flatten = Flatten()
        self.fc = Linear(4, 10)
    
    def call(self, x):
        x = self.stem(x)
        x = self.op(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = ConvNet()
model.build((None, 8, 8, 1))
# optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

loss_m = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function(jit_compile=True)
def train_step(batch):
    x, y = batch
    with tf.GradientTape() as tape:
        p = model(x, training=True)
        per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(y, p, from_logits=True)
        loss = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_m.update_state(per_example_loss)
    accuracy.update_state(y, p)

epochs = 200
for epoch in range(epochs):
    it = iter(ds_train)
    loss_m.reset_state()
    accuracy.reset_state()
    for step in range(steps_per_epoch):
        batch = next(it)
        train_step(batch)
    print("Epoch %d/%d" % (epoch + 1, epochs))
    print("Train - loss: %.4f, acc: %.4f" % (loss_m.result(), accuracy.result()))
    print(tf.nn.softmax(model.op.alpha))
