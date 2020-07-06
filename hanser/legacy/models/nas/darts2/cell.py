import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import ReLU

from hanser.models.layers2 import bn, conv2d
from hanser.models.legacy.layers import add, cat
from hanser.models.legacy.nas import NORMAL_OPS, REDUCTION_OPS, FactorizedReduce


class MixedOp(Model):

    def __init__(self, ops, channels, stride, name=None):
        super().__init__(name=name)
        self.ops = [
            op(channels, stride)
            for op in ops.values()
        ]

    def call(self, inputs):
        x, weights = inputs
        weights = tf.nn.softmax(weights)
        xs = [
            l(x) * weights[i]
            for i, l in enumerate(self.ops)
        ]
        return sum(xs)


def preprocess(channels, stride, name):
    if stride == 1:
        return Sequential([
            ReLU(name=name + '/relu'),
            conv2d(channels, 1, name=name + '/conv'),
            bn(affine=False, name=name + '/bn'),
        ], name=name)
    else:
        return FactorizedReduce(channels, name=name)


def normal_cell(s0, s1, weights, channels, num_units, name):
    assert num_units == len(weights)

    s0 = preprocess(channels, 2 if s0.shape[2] != s1.shape[2] else 1, name=name + '/preprocess0')(s0)
    s1 = preprocess(channels, 1, name=name + '/preprocess1')(s1)

    s = [s0, s1]
    for i in range(num_units):
        unit_name = name + ('/unit%d' % i)
        xs = []
        for j, x in enumerate(s):
            op = MixedOp(NORMAL_OPS, channels, 1, name=unit_name + ('/in%d' % (j - 2)))
            xs.append(op([x, weights[i]]))
        s.append(add(xs, name=unit_name + '/merge'))
    return cat(s[-num_units:], name=name + '/merge')


def reduction_cell(s0, s1, weights, channels, num_units, name):
    assert num_units == len(weights)

    s0 = preprocess(channels, 1, name=name + '/preprocess0')(s0)
    s1 = preprocess(channels, 1, name=name + '/preprocess1')(s1)

    s = [s0, s1]
    for i in range(num_units):
        unit_name = name + ('/unit%d' % i)
        xs = []
        for j, x in enumerate(s):
            op = MixedOp(REDUCTION_OPS, channels, stride=int(j < 2) + 1, name=unit_name + ('/in%d' % (j - 2)))
            xs.append(op([x, weights[i]]))
        s.append(add(xs, name=unit_name + '/merge'))
    return cat(s[-num_units:], name=name + '/merge')
