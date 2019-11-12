"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4

For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

The weights for all 16 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381) (CVPR 2018)

Tests comparing this model to the existing Tensorflow model can be
found at [mobilenet_v2_keras]
(https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""

import os
import warnings
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, BatchNormalization, ReLU, DepthwiseConv2D, Add
from tensorflow.keras.utils import get_file

from hanser.model.layers import conv2d, bn


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    input_size = K.int_shape(inputs)[1:3]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


# TODO Change path to v1.1
BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(alpha=1.0,
                input_shape=(224, 224, 3),
                output_stride=32,
                pretrained=True):
    """Instantiates the MobileNetV2 architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper, but the name is kept for
        consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid alpha, rows when
            weights='imagenet'
    """

    rows = input_shape[0]
    cols = input_shape[1]

    if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
        raise ValueError('If imagenet weights are being loaded, '
                         'alpha can be one of `0.35`, `0.50`, `0.75`, '
                         '`1.0`, `1.3` or `1.4` only.')

    if rows != cols or rows not in [96, 128, 160, 192, 224]:
        rows = 224
        warnings.warn('`input_shape` is undefined or non-square, '
                      'or `rows` is not in [96, 128, 160, 192, 224].'
                      ' Weights for input shape (224, 224) will be'
                      ' loaded as the default.')

    inputs = Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = ZeroPadding2D(padding=correct_pad(inputs, 3),
                      name='Conv1_pad')(inputs)
    x = conv2d(x, first_block_filters,
               kernel_size=3,
               stride=2,
               padding='valid',
               name='Conv1')
    x = bn(x, name='bn_Conv1')
    x = ReLU(6., name='Conv1_relu')(x)

    config = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    if output_stride == 16:
        config[5][3] = 1

    block_id = 0
    for t, c, n, s in config:
        x = _inverted_res_block(x, filters=c, alpha=alpha, stride=s,
                                expansion=t, block_id=block_id)
        block_id += 1
        for i in range(1, n):
            x = _inverted_res_block(x, filters=c, alpha=alpha, stride=1,
                                    expansion=t, block_id=block_id)
            block_id += 1

    last_block_filters = _make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280

    x = conv2d(x, last_block_filters,
               kernel_size=1,
               name='Conv_1')
    x = bn(x, name='Conv_1_bn')
    x = ReLU(6., name='out_relu')(x)

    model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    if pretrained:
        # Load weights.
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                      str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = conv2d(x, expansion * in_channels,
                   kernel_size=1,
                   name=prefix + 'expand')
        x = bn(x, name=prefix + 'expand_BN')
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                          name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride,
                        use_bias=False,
                        padding='same' if stride == 1 else 'valid',
                        name=prefix + 'depthwise')(x)
    x = bn(x, name=prefix + 'depthwise_BN')

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = conv2d(x, pointwise_filters,
               kernel_size=1,
               name=prefix + 'project')
    x = bn(x, name=prefix + 'project_BN')

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def mobilenetv2_backbone(input_shape, alpha=1.0, output_stride=32, pretrained=True):
    model = MobileNetV2(alpha, input_shape, output_stride, pretrained)
    if output_stride == 16:
        c3 = model.get_layer('block_6_expand_relu').output
        c4 = model.output
        cs = [c3, c4]
    elif output_stride == 32:
        c3 = model.get_layer('block_6_expand_relu').output
        c4 = model.get_layer('block_13_expand_relu').output
        c5 = model.output
        cs = [c3, c4, c5]
    else:
        raise ValueError('Invalid output_stride: %d' % output_stride)

    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = output_stride
    return backbone
