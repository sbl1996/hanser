"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""

import os

import numpy as np

import tensorflow as tf
from hanser.model.utils import decimate
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MaxPool2D, ZeroPadding2D, Flatten, Dense, Conv2D
from tensorflow.keras.utils import get_file

from hanser.model.layers import conv2d, bn, relu

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG16(input_shape, use_bn=True, pretrained=True):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

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
            x = MaxPool2D((2, 2), strides=(2, 2), name='block%d_pool' % block_id)(x)
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

    x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='vgg16')

    if pretrained:
        weights_path = get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            file_hash='64373286793e3c8b2b4e3219cbf3544b')
        model.load_weights(weights_path, by_name=True)

    return model


def vgg_backbone(input_shape, output_stride, pretrained=True):
    assert output_stride == 16
    model = VGG16(input_shape, pretrained)

    c3 = model.get_layer('block4_pool').input
    c4 = model.get_layer('block5_pool').input
    x = c4
    x = MaxPool2D((3, 3), strides=(1, 1), padding='same', name='block5_pool')(x)

    conv6 = Conv2D(1024, (3, 3),
                   dilation_rate=6,
                   padding='same',
                   name='conv6_conv')
    x = conv6(x)
    w, b = model.get_layer('fc1').get_weights()
    w = tf.reshape(w, [7, 7, 512, 4096])
    w = decimate(w, [3, 3, None, 4])
    b = tf.reshape(b, [4096])
    b = decimate(b, [4])
    conv6.set_weights([w, b])
    x = bn(x, name='conv6_bn')
    x = relu(x, name='conv6_relu')

    conv7 = Conv2D(1024, (1, 1),
                   name='conv7_conv')
    x = conv7(x)
    w, b = model.get_layer('fc2').get_weights()
    w = tf.reshape(w, [1, 1, 4096, 4096])
    w = decimate(w, [1, 1, 4, 4])
    b = tf.reshape(b, [4096])
    b = decimate(b, [4])
    conv7.set_weights([w, b])
    x = bn(x, name='conv7_bn')
    x = relu(x, name='conv7_relu')

    cs = [c3, x]
    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = output_stride
    return backbone


def format_h5(fp):
    import h5py
    f = h5py.File(fp, 'r+')
    with h5py.File(fp, 'r+') as f:
        f.attrs.pop('backend')
        f.attrs.pop('keras_version')
        for k in f.keys():
            if ('bn' in k) or ('relu' in k) or ('conv' not in k):
                del f[k]
        f.attrs['layer_names'] = np.array([
            s.encode()
            for s in f.keys()
        ])