"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""

import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MaxPool2D, ZeroPadding2D, Flatten, Dense, Conv2D
from tensorflow.keras.utils import get_file

from hanser.model.layers import conv2d, bn, relu

WEIGHTS_PATH = ('https://github.com/sbl1996/hanser/'
                'releases/download/0.1/'
                'vgg16_backbone.h5')


def VGG16(input_shape, use_bn=True, pretrained=True):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    inputs = Input(shape=input_shape, name='input')
    config = [ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

    x = inputs
    block_id = 1
    layer_id = 1
    for l in config:
        if l == 'M':
            x = MaxPool2D((2, 2), strides=(2, 2), padding='same', name='block%d_pool' % block_id)(x)
            block_id += 1
            layer_id = 0
        else:
            x = conv2d(x, l, (3, 3),
                       use_bias=True,
                       name='block%d_conv%d' % (block_id, layer_id))
            if use_bn:
                x = bn(x, name='block%d_bn%d' % (block_id, layer_id))
            x = relu(x, name='block%d_relu%d' % (block_id, layer_id))
        layer_id += 1

    x = MaxPool2D((3, 3), strides=(1, 1), padding='same', name='block5_pool')(x)

    x = conv2d(x, 1024, (3, 3),
               use_bias=True,
               dilation=6,
               name='conv6_conv')
    x = bn(x)
    x = relu(x)

    x = conv2d(x, 1024, (1, 1),
               use_bias=True,
               name='conv7_conv')
    x = bn(x)
    x = relu(x)

    model = Model(inputs, x, name='vgg16')

    if pretrained:
        weights_path = get_file(
            'vgg16_backbone.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='0d2d9c8759f9a5678c18d7c8788d1e66')
        model.load_weights(weights_path, by_name=True)

    return model


def vgg_backbone(input_shape, output_stride, pretrained=True):
    assert output_stride == 16
    model = VGG16(input_shape, pretrained=pretrained)

    c3 = model.get_layer('block4_pool').input
    c4 = model.output

    cs = [c3, c4]

    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = output_stride
    return backbone

