import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAvgPool2D, Layer

from hanser.models.legacy.layers import conv2d, bn, dense
from hanser.models.legacy.nas.darts import normal_cell, reduction_cell, NORMAL_OPS, REDUCTION_OPS


class Weight(Layer):

    def __init__(self, num_ops, num_op_types, **kwargs):
       self.num_ops = num_ops
       self.num_op_tyeps = num_op_types
       super().__init__(**kwargs)

    def build(self, input_shape):
        num_steps = self.num_ops - 3
        self.alpha = self.add_weight(
            shape=[num_steps, self.num_op_tyeps],
            initializer='zeros', dtype=tf.float32,
            trainable=True,
            name='alpha')

    def call(self, inputs=None):
       return self.alpha

    def get_config(self):
        return {'num_ops': self.num_ops, 'num_op_types': self.num_op_tyeps}


def get_weights(num_ops, num_op_types, name=None):
    w_init = tf.constant_initializer(0.)
    w = tf.Variable(w_init([num_ops - 3, num_op_types], tf.float32),
                    trainable=True,
                    name=name)
    return w


def darts(input_shape=(32, 32, 3), num_classes=10, start_channels=16, num_cells=(2, 3, 3), num_ops=7):
    channels = start_channels
    inputs = Input(input_shape, name='input')
    x = conv2d(inputs, start_channels * 3, kernel_size=3)
    x = bn(x)

    normal_weight = Weight(num_ops, len(NORMAL_OPS))(x)
    reduction_weight = Weight(num_ops, len(REDUCTION_OPS))(x)

    strides = [1, 2, 2]
    s0 = s1 = x
    for s, l in zip(strides, num_cells):
        if s == 1:
            s0, s1 = s1, normal_cell(s0, s1, normal_weight, channels, num_ops)
        else:
            channels *= 2
            s0, s1 = s1, reduction_cell(s0, s1, reduction_weight, channels, num_ops)
        for i in range(l - 1):
            s0, s1 = s1, normal_cell(s0, s1, normal_weight, channels, num_ops)

    x = GlobalAvgPool2D()(s1)
    x = dense(x, num_classes)
    model = Model(inputs=inputs, outputs=x)
    return model