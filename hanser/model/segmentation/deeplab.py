import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.layers import ReLU, Concatenate, AvgPool2D, UpSampling2D, Input, Lambda

from hanser.model.layers import conv2d, bn


def aspp(x, channels=256, rates=(6, 12, 18)):

    x1 = conv2d(x, channels, kernel_size=1)
    x1 = bn(x1)

    x2 = conv2d(x, channels, kernel_size=3, dilation=rates[0])
    x2 = bn(x2)

    x3 = conv2d(x, channels, kernel_size=3, dilation=rates[1])
    x3 = bn(x3)

    x4 = conv2d(x, channels, kernel_size=3, dilation=rates[2])
    x4 = bn(x4)

    im = AvgPool2D(x.shape[1:3])(x)
    im = conv2d(im, channels, kernel_size=1)
    im = bn(im)
    im = UpSampling2D(x.shape[1:3], interpolation='bilinear')(im)

    x = Concatenate()([x1, x2, x3, x4, im])
    x = ReLU()(x)
    x = conv2d(x, channels, kernel_size=1)
    x = bn(x)
    x = ReLU()(x)

    return x


def deeplabv3(backbone, num_classes):
    input_shape = backbone.input_shape[1:]
    inputs = Input(input_shape)

    x = backbone(inputs)
    x = aspp(x)
    logits = conv2d(x, num_classes, kernel_size=1, use_bias=True)
    logits = Lambda(tf.image.resize,
                    arguments=dict(size=input_shape[:2]),
                    name='upsampling_logits')(logits)
    model = Model(inputs=inputs, outputs=logits)
    return model
