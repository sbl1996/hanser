"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

Adapted from code contributed by BigMoyan.
"""

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

import tensorflow.keras.backend as K
from tensorflow.keras import utils
from tensorflow.keras.layers import ZeroPadding2D, Activation, Add, MaxPooling2D, Input
from tensorflow.keras.models import Model

from hanser.model.layers import conv2d, bn


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation_rate=(1, 1)):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv2d(input_tensor, filters1, 1, padding='valid',
               use_bias=True,
               dilation=dilation_rate,
               name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv2d(x, filters2, kernel_size,
               use_bias=True,
               dilation=dilation_rate,
               name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv2d(x, filters3, (1, 1), padding='valid',
               use_bias=True,
               dilation=dilation_rate,
               name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c')

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               dilation_rate=(1, 1)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv2d(input_tensor, filters1, (1, 1),
               stride=strides, padding='valid',
               use_bias=True,
               name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv2d(x, filters2, kernel_size,
               use_bias=True,
               dilation=dilation_rate,
               name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv2d(x, filters3, (1, 1), padding='valid',
               use_bias=True,
               dilation=dilation_rate,
               name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c')

    shortcut = conv2d(input_tensor, filters3, (1, 1),
                      stride=strides, padding='valid',
                      use_bias=True,
                      name=conv_name_base + '1')
    shortcut = bn(shortcut, name=bn_name_base + '1')

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape,
             pretrained=True,
             output_stride=32,
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pretrained: whether to load pretrained model.
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

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = conv2d(x, 64, (7, 7),
               stride=(2, 2),
               padding='valid',
               use_bias=True,
               name='conv1')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    if output_stride == 16:
        strides = (1, 1)
        dilation_rate = (2, 2)
    else:
        strides = (2, 2)
        dilation_rate = (1, 1)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides, dilation_rate=dilation_rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=dilation_rate)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=dilation_rate)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # Load weights.
    if pretrained:
        weights_path = utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)

    return model
