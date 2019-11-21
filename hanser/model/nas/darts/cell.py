import tensorflow as tf

from tensorflow.keras.layers import Softmax

from hanser.model.layers import relu, bn, conv2d, cat, add
from hanser.model.nas.darts.ops import factorized_reduce, NORMAL_OPS, REDUCTION_OPS


def mixed_normal_op(x, weights, ops, channels):
    assert len(ops) == weights.shape[0]
    s = [
        op(x, channels) * weights[i]
        for i, op in enumerate(ops.values())
    ]
    return add(s)


def mixed_reduction_op(x, weights, ops, channels, stride):
    assert len(ops) == weights.shape[0]
    s = [
        op(x, channels, stride) * weights[i]
        for i, op in enumerate(ops.values())
    ]
    x = add(s)
    return x


def preprocess(x, channels, stride):
    if stride == 1:
        x = relu(x)
        x = conv2d(x, channels, 1)
        x = bn(x, affine=False)
    else:
        x = factorized_reduce(x, channels)
    return x


def normal_cell(s0, s1, weights, channels, num_ops):
    num_steps = num_ops - 3  # Exclude input nodes and output node
    assert num_steps == weights.shape[0]

    weights = tf.math.softmax(weights, axis=-1)
    s0 = preprocess(s0, channels, 2 if s0.shape[2] != s1.shape[2] else 1)
    s1 = preprocess(s1, channels, 1)

    s = [s0, s1]
    for i in range(num_steps):
        s.append(add([mixed_normal_op(x, weights[i], NORMAL_OPS, channels) for x in s]))
    return cat(s[-num_steps:])


def reduction_cell(s0, s1, weights, channels, num_ops):
    num_steps = num_ops - 3  # Exclude input nodes and output node
    assert num_steps == weights.shape[0]

    weights = Softmax()(weights)
    s0 = preprocess(s0, channels, 1)
    s1 = preprocess(s1, channels, 1)

    s = [s0, s1]
    for i in range(num_steps):
        s.append(
            add([mixed_reduction_op(
                x, weights[i], REDUCTION_OPS, channels, stride=int(j < 2) + 1)
                for j, x in enumerate(s)]
            ))
    return cat(s[-num_steps:])
