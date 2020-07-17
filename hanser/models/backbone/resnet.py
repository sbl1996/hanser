from tensorflow.keras import Model
from tensorflow.keras.layers import Add, MaxPool2D, ZeroPadding2D, Input
from tensorflow.keras.utils import get_file

from hanser.models.functional.layers import conv2d, norm, act


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
                          bias=True, name=name + '_0_conv')
        shortcut = norm(shortcut, name=name + '_0_bn')
    else:
        shortcut = x

    x = conv2d(x, filters, 1, stride=stride,
               bias=True, name=name + '_1_conv')
    x = norm(x, name=name + '_1_bn')
    x = act(x, 'relu', name=name + '_1_relu')

    x = conv2d(x, filters, kernel_size, bias=True, dilation=dilation,
               name=name + '_2_conv')
    x = norm(x, name=name + '_2_bn')
    x = act(x, 'relu', name=name + '_2_relu')

    x = conv2d(x, 4 * filters, 1, bias=True,
               name=name + '_3_conv')
    x = norm(x, name=name + '_3_bn')

    x = Add(name=name + '_add')([shortcut, x])
    x = act(x, 'relu', name=name + '_out')
    return x


def stack1(x, filters, blocks, stride1=2, dilation=None, name=None):
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
    if isinstance(dilation, (tuple, list)):
        assert len(dilation) == blocks
    elif isinstance(dilation, int):
        dilation = [dilation] * blocks
    else:
        dilation = [1] * blocks
    x = block1(x, filters, stride=stride1, dilation=dilation[0], name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, dilation=dilation[i - 1], name=name + '_block' + str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           dilation=1, conv_shortcut=False, name=None):
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
    if stride == 2:
        dilation = 1
    preact = norm(x, name=name + '_preact_bn')
    preact = act(preact, 'relu', name=name + '_preact_relu')

    if conv_shortcut is True:
        shortcut = conv2d(preact, 4 * filters, 1, stride=stride, bias=True,
                          name=name + '_0_conv')
    else:
        shortcut = MaxPool2D(1, strides=stride)(x) if stride > 1 else x

    x = conv2d(preact, filters, 1, stride=1, bias=False,
               name=name + '_1_conv')
    x = norm(x, name=name + '_1_bn')
    x = act(x, 'relu', name=name + '_1_relu')

    # x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = conv2d(x, filters, kernel_size, stride=stride, dilation=dilation,
               name=name + '_2_conv')
    x = norm(x, name=name + '_2_bn')
    x = act(x, 'relu', name=name + '_2_relu')

    x = conv2d(x, 4 * filters, 1, bias=True,
               name=name + '_3_conv')
    x = Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, dilation=None, name=None):
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
    if isinstance(dilation, (tuple, list)):
        assert len(dilation) == blocks
    elif isinstance(dilation, int):
        dilation = [dilation] * blocks
    else:
        dilation = [1] * blocks
    x = block2(x, filters, conv_shortcut=True, dilation=dilation[0], name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, dilation=dilation[i - 1], name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, dilation=dilation[-1], name=name + '_block' + str(blocks))
    return x


def ResNet(input_shape,
           stack_fn,
           preact,
           model_name='resnet'):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the models is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, models name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the models.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the models will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the models will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras models instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = conv2d(x, 64, 7, stride=2, padding='valid', bias=True,
               name='conv1_conv')

    if preact is False:
        x = norm(x, name='conv1_bn')
        x = act(x, 'relu', name='conv1_relu')

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = MaxPool2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = norm(x, name='post_bn')
        x = act(x, 'relu', name='post_relu')

    inputs = img_input

    # Create models.
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
        model.load_weights(weights_path, by_name=True)


def ResNet50(input_shape, pretrained=True):
    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, 'resnet50')
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

    model = ResNet(input_shape, stack_fn, False, 'resnet101')
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

    model = ResNet(input_shape, stack_fn, False, 'resnet152')
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

    model = ResNet(input_shape, stack_fn, True, 'resnet50v2')
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

    model = ResNet(input_shape, stack_fn, True, 'resnet101v2')
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

    model = ResNet(input_shape, stack_fn, True, 'resnet152v2')
    if pretrained:
        load_weights(model, 'resnet152v2')

    return model


def get_resnetv2(depth, input_shape, output_stride=32, pretrained=True):
    if pretrained:
        assert depth in [50, 101, 152]
    config = {
        50: [
            [64, 3, 2, 1],
            [128, 4, 2, 1],
            [256, 6, 2, 1],
            [512, 3, 1, 1],
        ],
        101: [
            [64, 3, 2, 1],
            [128, 4, 2, 1],
            [256, 23, 2, 1],
            [512, 3, 1, 1],
        ],
        152: [
            [64, 3, 2, 1],
            [128, 8, 2, 1],
            [256, 36, 2, 1],
            [512, 3, 1, 1],
        ],
    }[depth]

    if output_stride == 16:
        config[-2][2] = 1
        config[-1][3] = 2

    name = 'resnet%dv2' % depth

    def stack_fn(x):
        for i, (c, n, s, d) in enumerate(config):
            x = stack2(x, c, n, stride1=s, name='conv%d' % (i + 2))
        return x

    model = ResNet(input_shape, stack_fn, True, name)
    if pretrained:
        load_weights(model, name)

    return model


def get_resnet(depth, input_shape, output_stride=32, pretrained=True):
    if pretrained:
        assert depth in [50, 101, 152]

    config = {
        50: [
            [64, 3, 1, 1],
            [128, 4, 2, 1],
            [256, 6, 2, 1],
            [512, 3, 2, 1],
        ],
        101: [
            [64, 3, 1, 1],
            [128, 4, 2, 1],
            [256, 23, 2, 1],
            [512, 3, 2, 1],
        ],
        152: [
            [64, 3, 1, 1],
            [128, 8, 2, 1],
            [256, 36, 2, 1],
            [512, 3, 2, 1],
        ],
    }[depth]

    name = 'resnet%d' % depth
    if output_stride == 16:
        config[-1][2] = 1
        config[-1][3] = 2

    def stack_fn(x):
        for i, (c, n, s, d) in enumerate(config):
            x = stack1(x, c, n, stride1=s, name='conv%d' % (i + 2))
        return x

    model = ResNet(input_shape, stack_fn, False, name)
    if pretrained:
        load_weights(model, name)

    return model


def resnet_backbone(output_stride, **kwargs):
    model = get_resnet(output_stride=output_stride, **kwargs)
    if output_stride == 16:
        c3 = model.get_layer('conv4_block1_1_conv').input
        c4 = model.output
        cs = [c3, c4]
    elif output_stride == 32:
        c3 = model.get_layer('conv4_block1_1_conv').input
        c4 = model.get_layer('conv5_block1_1_conv').input
        c5 = model.output
        cs = [c3, c4, c5]
    else:
        raise ValueError('Invalid output_stride: %d' % output_stride)

    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = output_stride
    return backbone


def resnetv2_backbone(output_stride, **kwargs):
    model = get_resnetv2(output_stride=output_stride, **kwargs)
    if output_stride == 16:
        c3 = model.get_layer('conv3_block4_2_conv').input
        c4 = model.output
        cs = [c3, c4]
    elif output_stride == 32:
        c3 = model.get_layer('conv3_block4_2_conv').input
        c4 = model.get_layer('conv4_block6_2_conv').input
        c5 = model.output
        cs = [c3, c4, c5]
    else:
        raise ValueError('Invalid output_stride: %d' % output_stride)

    backbone = Model(inputs=model.inputs, outputs=cs)
    backbone.output_stride = output_stride
    return backbone