from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Add, Input, GlobalAvgPool2D

from hanser.models.layers2 import bn, conv2d, act, dense


def conv_block(x, channels=16, kernel_size=3, strides=1,
               batch_normalization=False, activation=None, name='conv'):
    if batch_normalization:
        x = bn(name=name + '/bn')(x)
    if activation:
        x = act(name=name + '/relu')(x)

    x = conv2d(channels, kernel_size, strides, name=name + '/conv')(x)
    return x


def basic_block(x, channels=16, k=10, strides=1, dropout=None, conv_shortcut=False, name='block'):

    identity = x

    x = bn(name=name + "/bn1")(x)
    x = act(name=name + "/relu1")(x)
    if conv_shortcut:
        identity = x
    x = conv_block(x, channels * k, kernel_size=3, strides=strides, name=name + "/conv1")

    if dropout:
        x = Dropout(dropout, name=name + "/dropout")(x)

    x = conv_block(x, channels * k, kernel_size=3, strides=1,
                   batch_normalization=True, activation=True, name=name + "/conv2")

    if conv_shortcut:
        identity = conv_block(identity, channels * k, kernel_size=1, strides=strides, name=name + "/shortcut")

    x = Add(name=name + "/add")([x, identity])
    return x


def stack(x, num_blocks, channels, k, strides, dropout=None, name='stage'):
    x = basic_block(x, channels, k=k, strides=strides,
                    dropout=dropout, conv_shortcut=True, name=name + "/block1")
    for i in range(num_blocks - 1):
        x = basic_block(x, channels, k=k, strides=1,
                        dropout=dropout, conv_shortcut=False, name=name + "/block" + str(i + 2))
    return x


def wide_resnet(input_shape=(32, 32, 3), channels=16, depth=28, k=10, dropout=None, num_classes=10):
    num_blocks = int((depth - 4) / 6)

    inputs = Input(shape=input_shape)

    x = conv_block(inputs, channels, kernel_size=3, name='init_block')

    x = stack(x, num_blocks, channels * 1, k=k, strides=1, dropout=dropout, name='stage1')
    x = stack(x, num_blocks, channels * 2, k=k, strides=2, dropout=dropout, name='stage2')
    x = stack(x, num_blocks, channels * 4, k=k, strides=2, dropout=dropout, name='stage3')

    x = bn(name='post_act/bn')(x)
    x = act(name='post_act/relu')(x)

    x = GlobalAvgPool2D(name='pool')(x)
    outputs = dense(num_classes, name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
