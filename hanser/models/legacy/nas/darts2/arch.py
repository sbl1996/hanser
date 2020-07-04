import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, GlobalAvgPool2D, Layer

from hanser.models.legacy.layers import conv2d, bn, dense
from hanser.models.legacy.nas.darts2.cell import normal_cell, reduction_cell, NORMAL_OPS, REDUCTION_OPS


class Weight(Layer):

    def __init__(self, num_units, num_op_types, **kwargs):
        super().__init__(**kwargs)
        self.alphas = []
        self.num_edges = sum(range(2, 2 + num_units))
        self.num_op_tyeps = num_op_types

    def build(self, input_shape):
        for i in range(self.num_edges):
            self.alphas.append(
                self.add_weight(
                    shape=(self.num_op_tyeps,),
                    initializer='zeros', dtype=tf.float32,
                    trainable=True, name='alpha'
                )
            )

    def call(self, inputs):
        return self.alphas

    def get_config(self):
        config = super().get_config()
        config.update({'num_units': self.num_units, 'num_op_types': self.num_op_tyeps})
        return config


def darts(input_shape=(32, 32, 3), num_classes=10, start_channels=16, num_cells=(2, 3, 3), num_units=4):
    channels = start_channels
    inputs = Input(input_shape, name='input')
    x = conv2d(inputs, start_channels * 3, kernel_size=3, name='stem/conv')
    x = bn(x, name='stem/bn')

    normal_weights = Weight(num_units, len(NORMAL_OPS), name='normal_alpha')(x)
    reduction_weights = Weight(num_units, len(REDUCTION_OPS), name='reduction_alpha')(x)

    strides = [1, 2, 2]
    s0 = s1 = x
    for i, (s, l) in enumerate(zip(strides, num_cells)):
        name = 'stage%d' % i
        if s == 1:
            s0, s1 = s1, normal_cell(s0, s1, normal_weights, channels, num_units, name=name + '/cell0')
        else:
            channels *= 2
            s0, s1 = s1, reduction_cell(s0, s1, reduction_weights, channels, num_units, name=name + '/cell0')
        for i in range(1, l):
            s0, s1 = s1, normal_cell(s0, s1, normal_weights, channels, num_units, name=name + ('/cell%d' % i))

    x = GlobalAvgPool2D(name='final_pool')(s1)
    x = dense(x, num_classes, name='fc')
    model = Model(inputs=inputs, outputs=x)
    return model
