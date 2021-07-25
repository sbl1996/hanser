import numpy as np
import tensorflow as tf

def zero_ops(layer, input_shape=None, output_shape=None):
    return 0


def count_conv(layer, h, w):
    if isinstance(layer, tf.keras.Sequential):
        for l in layer.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                layer = l
                break
    total_ops = h * w * np.prod(layer.kernel.shape) / np.prod(layer.strides)
    return total_ops


def count_linear(layer):
    total_ops = np.prod(layer.kernel.shape)
    return total_ops


def count_activation(layer, input_shape=None, output_shape=None):
    input_shape = input_shape or layer.input_shape[1:]
    output_shape = output_shape or layer.output_shape[1:]
    total_ops = np.prod(input_shape)
    return total_ops