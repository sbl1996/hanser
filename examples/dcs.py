import math

from toolz import curry

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import metrics as M
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from torchvision.transforms import RandomCrop

from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare
from hanser.datasets.cifar import load_cifar10
from hanser.transform import random_crop, cutout, normalize, to_tensor

from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import Genotype
from hanser.models.layers import set_defaults
from hanser.train.trainer import Trainer
from hanser.train.lr_schedule import CosineLR
from hanser.losses import CrossEntropy


@curry
def transform(input, training):

    input = normalize(input, [0.49716908, 0.50092334], [0.05410922, 0.05399005])

    # image = tf.cast(image, tf.bfloat16)

    target = input
    return input, target

feedback_bits = 128

data = np.load("~/Downloads/Hdata_sub.npy")
x_train, x_test = train_test_split(data, test_size=0.1, random_state=42)

mul = 1
num_train_examples = len(x_train)
num_test_examples = len(x_test)
batch_size = 32 * mul
eval_batch_size = batch_size * 2
steps_per_epoch = num_train_examples // batch_size
test_steps = math.ceil(num_test_examples / eval_batch_size)

ds = tf.data.Dataset.from_tensor_slices((x_train,))
ds_test = tf.data.Dataset.from_tensor_slices((x_test,))

ds_train = prepare(ds, batch_size, transform(training=True), training=True, buffer_size=10000)
ds_test = prepare(ds_test, eval_batch_size, transform(training=False), training=False)

strategy = get_colab_tpu()
if strategy:
    policy = mixed_precision.Policy('mixed_bfloat16')
    mixed_precision.set_policy(policy)
    tf.distribute.experimental_set_strategy(strategy)

    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)


def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config



def Encoder(x,feedback_bits):
    B=4
    with tf.compat.v1.variable_scope('Encoder'):
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output
def Decoder(x,feedback_bits):
    B=4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = layers.Dense(1024, activation='sigmoid')(x)
    x_ini = layers.Reshape((16, 32, 2))(x)

    for i in range(3):
        x = layers.Conv2D(8, 3, padding = 'SAME',activation="relu")(x_ini)
        x = layers.Conv2D(16,3, padding = 'SAME',activation="relu")(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        x_ini = keras.layers.Add()([x_ini, x])


    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x_ini)

    return decoder_output


set_defaults({
    'weight_decay': 3e-4
})
drop_path = 0.2
# model = DARTS(36, 20, True, drop_path, 10, PDARTS)
model = NASNet(4, 5, True, drop_path, 10, PDARTS)
model.build((None, 32, 32, 3))
# model.call(tf.keras.layers.Input((32, 32, 3)))
# model.summary()

criterion = CrossEntropy(auxiliary_weight=0.4)
model.fit
base_lr = 0.025
epochs = 600
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=10)
optimizer = SGD(lr_shcedule, momentum=0.9, nesterov=True)
metrics = [
    M.Mean(name='loss'), M.CategoricalAccuracy(name='acc')]
test_metrics = [
    M.CategoricalCrossentropy(name='loss', from_logits=True), M.CategoricalAccuracy(name='acc')]
metric_transform = lambda x: x[0]

trainer = Trainer(model, criterion, optimizer, metrics, test_metrics,
                  grad_clip_norm=5.0, metric_transform=metric_transform,
                  multiple_steps=True)


class DropPathRateSchedule(Callback):

    def on_epoch_begin(self, epoch, logs=None):
        rate = epoch / epochs * drop_path
        for l in model.submodules:
            if 'drop' in l.name:
                l.rate = rate


trainer.fit(epochs, ds_train, steps_per_epoch, ds_test, test_steps, val_freq=5,
            callbacks=[DropPathRateSchedule()])
