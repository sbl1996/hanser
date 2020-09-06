import math

from toolz import curry

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.metrics import MeanMetricWrapper

import tensorflow_addons as tfa

from hanser.tpu import get_colab_tpu
from hanser.datasets import prepare

from hanser.models.layers import Conv2d, Linear
from hanser.models.layers import set_defaults
from hanser.train.lr_schedule import CosineLR
from hanser.transform import normalize


def Num2Bit(Num, B):
    Num_ = tf.cast(Num, tf.int32)

    def integer2bit(integer, num_bits=B * 2):
        exponent_bits = tf.range((num_bits - 1), -1, -1, dtype=tf.int32)
        out = tf.expand_dims(integer, -1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = tf.reshape(bit[:, :, B:], [-1, Num_.shape[1] * B])
    return tf.cast(bit, tf.float32)


def Bit2Num(Bit, B):
    Bit_ = tf.cast(Bit, tf.float32)
    Bit_ = tf.reshape(Bit_, [-1, Bit_.shape[1] // B, B])
    exponent_bits = tf.range((B * 2 - 1), -1, -1, dtype=tf.int32)[B:]
    num = tf.reduce_sum(Bit_ * tf.cast(2 ** exponent_bits, tf.float32), -1)
    return num


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = Num2Bit(result, B)

    def custom_grad(dy):
        grad = tf.reduce_sum(tf.reshape(dy, [tf.shape(dy)[0], -1, B]), axis=2)
        return grad, None

    return result, custom_grad


@tf.custom_gradient
def DequantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    out = Bit2Num(x, B)
    result = (out + 0.5) / step

    def custom_grad(dy):
        grad_bit = tf.tile(tf.expand_dims(dy, -1), [1, 1, B])
        grad = tf.reshape(grad_bit, [tf.shape(grad_bit)[0], -1])
        return grad, None

    return result, custom_grad


class QuantizationLayer(Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__(**kwargs)

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


class DeuantizationLayer(Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__(**kwargs)

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


class Encoder(Model):
    B = 4

    def __init__(self, feedback_bits):
        super().__init__()
        self.conv = Sequential([
            Conv2d(2, 2, kernel_size=7, norm='def', act='def'),
            Conv2d(2, 2, kernel_size=7, norm='def', act='def'),
            Flatten(),
            Linear(1024, feedback_bits // self.B, act='sigmoid')
        ])
        self.quantization = QuantizationLayer(self.B)

    def call(self, x):
        x = self.conv(x)
        return self.quantization(x)


class ResBlock(Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, 8, kernel_size=7, norm='def', act='def')
        self.conv2 = Conv2d(8, 16, kernel_size=5, norm='def', act='def')
        self.conv3 = Conv2d(16, out_channels, kernel_size=3, norm='def', act='tanh')

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + identity


class Decoder(Model):
    B = 4

    def __init__(self, feedback_bits):
        super().__init__()

        self.dequantization = DeuantizationLayer(self.B)
        self.fc = Linear(feedback_bits // self.B, 1024)
        self.conv = Conv2d(2, 2, kernel_size=7, norm='def', act='sigmoid')
        self.res = Sequential([
            ResBlock(2, 2)
            for i in range(3)
        ])
        self.conv = Conv2d(2, 2, kernel_size=3, act='sigmoid')

    def call(self, x):
        x = self.dequantization(x)
        x = self.fc(x)
        x = tf.reshape(x, [-1, 16, 32, 2])
        x = self.res(x)
        x = self.conv(x)
        return x


class AutoEncoder(Model):

    def __init__(self, feedback_bits=128):
        super().__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def nmse(x, x_hat):
    b = tf.shape(x)[0]
    x_real = tf.reshape(x[:, :, :, 0], (b, -1))
    x_imag = tf.reshape(x[:, :, :, 1], (b, -1))
    x_hat_real = tf.reshape(x_hat[:, :, :, 0], (b, -1))
    x_hat_imag = tf.reshape(x_hat[:, :, :, 1], (b, -1))
    x_real = tf.cast(x_real, tf.complex64)
    x_imag = tf.cast(x_imag, tf.complex64)
    x_hat_real = tf.cast(x_hat_real, tf.complex64)
    x_hat_imag = tf.cast(x_hat_imag, tf.complex64)
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = tf.reduce_sum(tf.abs(x_C) ** 2, axis=1)
    mse = tf.reduce_sum(tf.abs(x_C - x_hat_C) ** 2, axis=1)
    result = mse / power
    return result


@curry
def transform(input, training):
    target = input
    return input, target


data = np.load("/Users/hrvvi/Downloads/Hdata_sub.npy")
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
    # policy = mixed_precision.Policy('mixed_bfloat16')
    # mixed_precision.set_policy(policy)
    tf.distribute.experimental_set_strategy(strategy)

    ds_train_dist = strategy.experimental_distribute_dataset(ds_train)
    ds_test_dist = strategy.experimental_distribute_dataset(ds_test)

set_defaults({
    'weight_decay': 3e-4
})
model = AutoEncoder(128)
model.build((None, 16, 32, 2))


class NMSE(MeanMetricWrapper):

    def __init__(self, name='nmse', dtype=None):
        super().__init__(nmse, name, dtype=dtype)


base_lr = 1e-3
# base_wd = 1e-4
epochs = 600
lr_shcedule = CosineLR(base_lr * mul, steps_per_epoch, epochs=epochs,
                       min_lr=0, warmup_min_lr=base_lr, warmup_epoch=10)
# wd_schedule = lambda: lr_shcedule(optimizer.iterations) / base_lr * base_wd
# optimizer = tfa.optimizers.AdamW(wd_schedule, lr_shcedule)
optimizer = Adam(lr_shcedule)
model.compile(optimizer=optimizer, loss='mse', metrics=[NMSE()])
model.fit(ds_train, epochs=100, steps_per_epoch=steps_per_epoch,
          validation_data=ds_test, validation_steps=test_steps, verbose=2)
