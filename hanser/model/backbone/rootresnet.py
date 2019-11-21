import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, MaxPool2D, Input, AvgPool2D

from hanser.model.layers import conv2d, bn, relu, PadChannel


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
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
        shortcut = conv2d(x, filters, 1, stride=stride,
                          name=name + '_0_conv')
        shortcut = bn(shortcut, name=name + '_0_bn')
    else:
        shortcut = x

    x = conv2d(x, filters, kernel_size, stride=stride,
               name=name + '_1_conv')
    x = bn(x, name=name + '_1_bn')
    x = relu(x, name=name + '_1_relu')

    x = conv2d(x, filters, kernel_size,
               name=name + '_2_conv')
    x = bn(x, name=name + '_2_bn')
    x = relu(x, name=name + '_2_relu')

    x = Add(name=name + '_add')([shortcut, x])
    x = relu(x, name=name + '_out')
    return x


def shortcut(x, out_channels, stride=1):
    if stride == 2:
        x = AvgPool2D()(x)
    x = PadChannel(out_channels)(x)
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
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
    if stride == 2:
        shortcut = AvgPool2D(padding='same', name=name + '_0_pool')(x)
        shortcut = PadChannel(filters, name=name + '_0_pad')(shortcut)
    else:
        shortcut = x

    x = bn(x, name=name + '_1_bn')
    x = conv2d(x, filters, kernel_size,
               name=name + '_1_conv')

    x = bn(x, name=name + '_2_bn')
    x = relu(x, name=name + '_2_relu')
    x = conv2d(x, filters, kernel_size, stride=stride,
               name=name + '_2_conv')

    x = Add(name=name + '_add')([shortcut, x])
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
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
    x = block2(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block2(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def RootResNet(input_shape,
               stack_fn,
               model_name='root_resnet'):

    inputs = Input(shape=input_shape)

    x = conv2d(inputs, 64, 3,
               name='conv1_conv1')
    x = bn(x, name='conv1_bn1')
    x = relu(x, name='conv1_relu1')
    x = conv2d(x, 64, 3,
               name='conv1_conv2')
    x = bn(x, name='conv1_bn2')
    x = relu(x, name='conv1_relu2')
    x = conv2d(x, 64, 3,
               name='conv1_conv3')
    x = bn(x, name='conv1_bn3')
    x = relu(x, name='conv1_relu3')

    x = MaxPool2D(3, strides=2, padding='same', name='pool1_pool')(x)

    x = stack_fn(x)

    # Create model.
    model = Model(inputs, x, name=model_name)
    return model


def get_root_resnet(depth, input_shape):
    config = {
        18: [
            [64, 2, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 2, 2],
        ],
        34: [
            [64, 4, 1],
            [128, 4, 2],
            [256, 4, 2],
            [512, 4, 2],
        ],
    }[depth]

    name = 'root_resnet%d' % depth

    def stack_fn(x):
        for i, (c, n, s) in enumerate(config):
            x = stack1(x, c, n, stride1=s, name='conv%d' % (i + 2))
        return x

    model = RootResNet(input_shape, stack_fn, name)

    return model


def root_resnet_backbone(depth, input_shape):
    model = get_root_resnet(depth, input_shape)
    if depth == 18:
        c3 = model.get_layer('conv4_block2_add').output
    elif depth == 34:
        c3 = model.get_layer('conv4_block4_add').output
    else:
        raise ValueError('Invalid depth: %d' % depth)
    c4 = model.output
    cs = [c3, c4]

    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = 16
    return backbone

