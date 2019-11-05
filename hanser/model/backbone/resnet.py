import os

import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, MaxPool2D, ZeroPadding2D, Reshape, Lambda, Input, GlobalAvgPool2D, Dense
from tensorflow.keras.utils import get_file

from hanser.model.layers import conv2d, bn, relu, dwconv2d, dense

BASE_WEIGHTS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101': ('34fb605428fcc7aa4d62f44404c11509',
                   '0f678c91647380debd923963594981b3')
}

def get_same_padding(kernel_size, stride, dilation):
    assert dilation > 1 and stride == 2
    k = kernel_size + (kernel_size - 1) * (dilation - 1)
    p = k - 1
    pl = p // 2
    pr = p - pl
    return pl, pr


def block1(x, filters, kernel_size=3, stride=1,
           dilation=1, conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """

    if conv_shortcut is True:
        shortcut = conv2d(x, 4 * filters, 1, stride=stride,
                          padding='valid',
                          use_bias=True,
                          name=name + '_0_conv')
        shortcut = bn(shortcut, name=name + '_0_bn')
    else:
        shortcut = x

    x = conv2d(x, filters, 1, stride=stride,
               padding='valid',
               use_bias=True,
               name=name + '_1_conv')
    x = bn(x, name=name + '_1_bn')
    x = relu(x, name=name + '_1_relu')

    x = conv2d(x, filters, kernel_size,
               use_bias=True,
               dilation=dilation,
               name=name + '_2_conv')
    x = bn(x, name=name + '_2_bn')
    x = relu(x, name=name + '_2_relu')

    x = conv2d(x, 4 * filters, 1,
               use_bias=True,
               padding='valid',
               name=name + '_3_conv')
    x = bn(x, name=name + '_3_bn')

    x = Add(name=name + '_add')([shortcut, x])
    x = relu(x, name=name + '_out')
    return x


def stack1(x, filters, blocks, stride1=2, dilation=1, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, dilation=dilation, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, dilation=dilation, name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """

    preact = bn(x, name=name + '_preact_bn')
    preact = relu(preact, name=name + '_preact_relu')

    if conv_shortcut is True:
        shortcut = conv2d(preact, 4 * filters, 1, stride=stride,
                          padding='valid',
                          use_bias=True,
                          name=name + '_0_conv')
    else:
        shortcut = MaxPool2D(1, strides=stride)(x) if stride > 1 else x

    x = conv2d(preact, filters, 1, stride=1,
               padding='valid',
               use_bias=True,
               name=name + '_1_conv')
    x = bn(x, name=name + '_1_bn')
    x = relu(x, name=name + '_1_relu')

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = conv2d(x, filters, kernel_size, stride=stride,
               padding='valid',
               name=name + '_2_conv')
    x = bn(x, name=name + '_2_bn')
    x = relu(x, name=name + '_2_relu')

    x = conv2d(x, 4 * filters, 1,
               padding='valid',
               use_bias=True,
               name=name + '_3_conv')
    x = Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """

    if conv_shortcut is True:
        shortcut = conv2d(x, (64 // groups) * filters, 1, stride=stride,
                          padding='valid',
                          name=name + '_0_conv')
        shortcut = bn(shortcut, name=name + '_0_bn')
    else:
        shortcut = x

    x = conv2d(x, filters, 1,
               padding='valid',
               name=name + '_1_conv')(x)
    x = bn(x, name=name + '_1_bn')
    x = relu(x, name=name + '_1_relu')

    c = filters // groups
    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = dwconv2d(x, kernel_size, stride=stride,
                 depth_multiplier=c,
                 padding='valid',
                 name=name + '_2_conv')
    x_shape = K.int_shape(x)[1:-1]
    x = Reshape(x_shape + (groups, c, c))(x)
    x = Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]),
               name=name + '_2_reduce')(x)
    x = Reshape(x_shape + (filters,))(x)
    x = bn(x, name=name + '_2_bn')
    x = relu(x, name=name + '_2_relu')

    x = conv2d(x, (64 // groups) * filters, 1,
               padding='valid',
               name=name + '_3_conv')(x)
    x = bn(x, name=name + '_3_bn')

    x = Add(name=name + '_add')([shortcut, x])
    x = relu(x, name=name + '_out')
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x


def ResNet(input_shape,
           stack_fn,
           preact,
           use_bias,
           model_name='resnet'):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
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

    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = conv2d(x, 64, 7, stride=2,
               padding='valid',
               use_bias=use_bias,
               name='conv1_conv')

    if preact is False:
        x = bn(x, name='conv1_bn')
        x = relu(x, name='conv1_relu')

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPool2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = bn(x, name='post_bn')
        x = relu(x, name='post_relu')

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)
    return model


def load_weights(model, model_name):
    if model_name in WEIGHTS_HASHES:
        file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(file_name,
                                BASE_WEIGHTS_PATH + file_name,
                                cache_subdir='models',
                                file_hash=file_hash)
        model.load_weights(weights_path)


def ResNet50(input_shape, pretrained=True, output_stride=32):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        if output_stride == 16:
            x = stack1(x, 512, 3, stride1=1, dilation=2, name='conv5')
        else:
            x = stack1(x, 512, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, True, 'resnet50')
    if pretrained:
        load_weights(model, 'resnet50')

    return model


def ResNet101(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 23, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, True, 'resnet101')
    if pretrained:
        load_weights(model, 'resnet101')

    return model


def ResNet152(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, True, 'resnet152')
    if pretrained:
        load_weights(model, 'resnet152')

    return model


def ResNet50V2(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 6, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, True, True, 'resnet50v2')
    if pretrained:
        load_weights(model, 'resnet50v2')

    return model


def ResNet101V2(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 23, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, True, True, 'resnet101v2')
    if pretrained:
        load_weights(model, 'resnet101v2')

    return model


def ResNet152V2(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 8, name='conv3')
        x = stack2(x, 256, 36, name='conv4')
        x = stack2(x, 512, 3, stride1=1, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, True, True, 'resnet152v2')
    if pretrained:
        load_weights(model, 'resnet152v2')

    return model


def ResNeXt50(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, False, 'resnext50')
    if pretrained:
        load_weights(model, 'resnext50')

    return model


def ResNeXt101(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 23, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, False, 'resnext101')
    if pretrained:
        load_weights(model, 'resnext101')

    return model
