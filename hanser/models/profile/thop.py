import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, Activation, Dense
from hanser.models.bn import BatchNormalization, SyncBatchNormalization
from hanser.models.layers import GlobalAvgPool
from hanser.models.pooling import AveragePooling2D, MaxPooling2D


def get_input_shape(layer, default_input_shape):
    try:
        return layer.input_shape[1:]
    except AttributeError:
        return default_input_shape


def get_output_shape(layer):
    try:
        return layer.output_shape[1:]
    except AttributeError:
        return None


def zero_ops(layer: Layer, input_shape):
    return 0


def count_conv(layer: Layer, input_shape):
    input_shape = layer.input_shape[1:]
    output_shape = layer.output_shape[1:]

    kernel_ops = np.prod(layer.kernel.shape[:2])
    bias_ops = 1 if layer.use_bias else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = np.prod(output_shape) * (layer.kernel.shape[2] * kernel_ops + bias_ops)

    return total_ops


def count_bn(layer: Layer, input_shape):
    input_shape = layer.input_shape[1:]
    output_shape = layer.output_shape[1:]

    total_ops = np.prod(input_shape) * 2
    return total_ops


def count_relu(layer: Layer, input_shape):
    input_shape = get_input_shape(layer, input_shape)

    total_ops = np.prod(input_shape)
    return total_ops


def count_avgpool(layer: Layer, input_shape):
    input_shape = layer.input_shape[1:]

    total_ops = np.prod(input_shape)
    return total_ops


def count_global_avgpool(layer: Layer, input_shape):
    input_shape = layer.input_shape[1:]

    kernel = input_shape[:2]
    total_add = np.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    total_ops = kernel_ops * input_shape[-1]
    return total_ops


def count_linear(layer: Layer, input_shape):
    input_shape = layer.input_shape[1:]
    output_shape = layer.output_shape[1:]

    total_mul = input_shape[-1]
    total_ops = total_mul * np.prod(output_shape)
    return total_ops


def count_sigmoid(layer: Layer, input_shape=None):
    input_shape = get_input_shape(layer, input_shape)

    total_ops = 19 * np.prod(input_shape)
    return total_ops


def count_swish(layer: Layer, input_shape):
    input_shape = get_input_shape(layer, input_shape)

    total_ops = 20 * np.prod(input_shape)
    return total_ops


def count_activation(layer: tf.keras.layers.Activation, input_shape):
    activation_fn = layer.activation
    activation_name = activation_fn.__name__
    if activation_name == 'relu':
        return count_relu(layer, input_shape)
    elif activation_name == 'sigmoid':
        return count_sigmoid(layer, input_shape)
    elif activation_name == 'swish':
        return count_swish(layer, input_shape)
    else:
        raise ValueError("Not supported: %s" % activation_name)


register_hooks = {
    Conv2D: count_conv,

    BatchNormalization: count_bn,
    SyncBatchNormalization: count_bn,

    Activation: count_activation,

    MaxPooling2D: zero_ops,
    AveragePooling2D: count_avgpool,

    GlobalAvgPool: count_global_avgpool,

    Dense: count_linear,
}


layer_table = {
    k: 0 for k in register_hooks
}


def count_mac(layer: Layer, input_shape=None):
    if isinstance(layer, (Model, Sequential)):
        total = 0
        input_shape = layer.layers[0].input_shape
        for l in layer.layers:
            total += count_mac(l, input_shape)
            input_shape = get_output_shape(l)
        return total
    elif type(layer) in register_hooks:
        typ = type(layer)
        ops = register_hooks[typ](layer, input_shape)
        layer_table[typ] += ops
        return ops
    else:
        total = 0
        for l in layer._flatten_layers(recursive=False, include_self=False):
            total += count_mac(l, input_shape)
            input_shape = get_output_shape(l)
        return total


def profile(model):
    for k in layer_table:
        layer_table[k] = 0
    assert isinstance(model, (Model, Sequential))
    input_shape = model.layers[0].input_shape[1:]
    model.call(tf.keras.layers.Input(input_shape))
    n = count_mac(model)
    t = {k: v for k, v in layer_table.items() if v != 0}
    t = {k: v / n for k, v in t.items()}
    return n, t