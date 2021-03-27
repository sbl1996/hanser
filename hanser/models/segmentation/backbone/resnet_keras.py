from tensorflow.keras import Model
from tensorflow.keras.layers import Add, MaxPool2D, ZeroPadding2D, Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.utils import get_file


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


def conv2d(x, channels, kernel_size, stride=1, padding='SAME', bias=True,
           dilation=1, name=None):
    return Conv2D(channels, kernel_size, stride, padding,
                  dilation_rate=dilation, use_bias=bias, name=name)(x)


def norm(x, name=None):
    return BatchNormalization(epsilon=1.001e-5, name=name)(x)


def act(x, type, name=None):
    return Activation(type, name=name)(x)


def block1(x, filters, kernel_size=3, stride=1,
           dilation=1, conv_shortcut=True, name=None):

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


def stack1(x, filters, blocks, stride=2, dilations=None, name=None):

    if dilations is None:
        dilations = [1] * blocks
    x = block1(x, filters, stride=stride, dilation=dilations[0], name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, dilation=dilations[i - 1], name=name + '_block' + str(i))
    return x


def ResNet(input_shape,
           stack_fn,
           preact,
           model_name='resnet'):

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
        x = stack1(x, 64, 3, stride=1, name='conv2')
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
        x = stack1(x, 64, 3, stride=1, name='conv2')
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
        x = stack1(x, 64, 3, stride=1, name='conv2')
        x = stack1(x, 128, 8, name='conv3')
        x = stack1(x, 256, 36, name='conv4')
        x = stack1(x, 512, 3, name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, 'resnet152')
    if pretrained:
        load_weights(model, 'resnet152')

    return model


def get_resnet(depth, output_stride=32, multi_grad=(1, 1, 1), input_shape=(224, 224, 3), pretrained=True):
    if pretrained:
        assert depth in [50, 101, 152]

    config = {
        50: [
            [64, 3, 1],
            [128, 4, 2],
            [256, 6, 2],
            [512, 3, 2],
        ],
        101: [
            [64, 3, 1],
            [128, 4, 2],
            [256, 23, 2],
            [512, 3, 2],
        ],
        152: [
            [64, 3, 1],
            [128, 8, 2],
            [256, 36, 2],
            [512, 3, 2],
        ],
    }[depth]

    name = 'resnet%d' % depth
    if output_stride == 16:
        config[-1][2] = 1

    def stack_fn(x):
        for i, (c, n, s) in enumerate(config):
            if i == 3 and s == 1:
                dilations = [2 * m for m in multi_grad]
                dilations[0] = 1
            else:
                dilations = None
            x = stack1(x, c, n, stride=s, dilations=dilations, name='conv%d' % (i + 2))
        return x

    model = ResNet(input_shape, stack_fn, False, name)
    if pretrained:
        load_weights(model, name)
    return model


def resnet_backbone(depth, output_stride=16, multi_grad=(1, 2, 4), **kwargs):
    model = get_resnet(depth=depth, output_stride=output_stride, multi_grad=multi_grad, **kwargs)
    c2 = model.get_layer('conv3_block1_1_conv').input
    c3 = model.get_layer('conv4_block1_1_conv').input
    c4 = model.get_layer('conv5_block1_1_conv').input
    c5 = model.output
    backbone = Model(inputs=model.inputs, outputs=[c2, c3, c4, c5])
    backbone.output_stride = output_stride
    backbone.feat_channels = [256, 512, 1024, 2048]
    return backbone


def resnet50(**kwargs):
    return resnet_backbone(50, **kwargs)

def resnet101(**kwargs):
    return resnet_backbone(101, **kwargs)

def resnet152(**kwargs):
    return resnet_backbone(152, **kwargs)