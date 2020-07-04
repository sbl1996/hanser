from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Input, GlobalAvgPool2D
from hanser.models.layers2 import bn, conv2d, act, dense


def conv_block(x, channels=16, kernel_size=3, strides=1,
               batch_normalization=False, activation=None, name='conv'):
    x = conv2d(channels, kernel_size, strides, name=name + '/conv')(x)
    if batch_normalization:
        x = bn(name=name + '/bn')(x)
    if activation:
        x = act(name=name + '/relu')(x)
    return x


def basic_block(x, channels=16, strides=1, conv_shortcut=False, name='block'):

    identity = x

    x = conv_block(x, channels, kernel_size=3, strides=strides,
                   batch_normalization=True, activation=True, name=name + "/conv1")
    x = conv_block(x, channels, kernel_size=3, strides=1,
                   batch_normalization=True, activation=True, name=name + "/conv2")

    if conv_shortcut:
        identity = conv_block(identity, channels, kernel_size=1, strides=strides,
                              batch_normalization=True, name=name + "/shortcut")

    x = Add(name=name + "/add")([x, identity])
    return x


def stack(x, num_blocks, channels, strides, name='stage'):
    x = basic_block(x, channels, strides=strides, conv_shortcut=True, name=name + "/block1")
    for i in range(num_blocks - 1):
        x = basic_block(x, channels, strides=1, conv_shortcut=False, name=name + "/block" + str(i + 2))
    return x


def resnet(input_shape=(32, 32, 3), channels=16, depth=110, num_classes=10):
    num_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)

    x = conv_block(inputs, channels, kernel_size=3, batch_normalization=True, activation=True,
                   name='init_block')

    x = stack(x, num_blocks, channels * 1, strides=1, name='stage1')
    x = stack(x, num_blocks, channels * 2, strides=2, name='stage2')
    x = stack(x, num_blocks, channels * 4, strides=2, name='stage3')

    x = GlobalAvgPool2D(name='pool')(x)
    outputs = dense(num_classes, name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
