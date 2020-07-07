import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, AvgPool2D, UpSampling2D, Input, Lambda

from hanser.legacy.models.layers import conv2d, bn, relu
from hanser.legacy.models.backbone.resnet import stack1, load_weights, ResNet


def ASPP(x, channels=256, rates=(6, 12, 18)):
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
    x = relu(x)
    x = conv2d(x, channels, kernel_size=1)
    x = bn(x)
    x = relu(x)

    return x


def get_resnet(model_name, input_shape, pretrained=True, output_stride=32, multi_grad=(1, 1, 1)):
    assert model_name in ['resnet50', 'resnet101']

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, {'resnet50': 6, 'resnet101': 23}[model_name], name='conv4')
        dilation = 2
        if output_stride == 16:
            dilation = tuple(d * dilation for d in multi_grad)
        x = stack1(x, 512, 3,
                   stride1=1 if output_stride == 16 else 2,
                   dilation=dilation,
                   name='conv5')
        return x

    model = ResNet(input_shape, stack_fn, False, True, model_name)
    if pretrained:
        load_weights(model, model_name)

    return model


def deeplabv3(input_shape, backbone, output_stride, multi_grad=(1, 1, 1), aspp=True, num_classes=21):
    assert backbone in ['resnet50', 'resnet101']
    backbone = get_resnet(backbone, input_shape, output_stride=output_stride, multi_grad=multi_grad)

    inputs = Input(input_shape)
    x = backbone(inputs)
    if aspp:
        x = ASPP(x)
    logits = conv2d(x, num_classes, kernel_size=1, use_bias=True)
    logits = Lambda(tf.compat.v1.image.resize_bilinear,
                    arguments=dict(
                        size=input_shape[:2],
                        align_corners=True,
                    ),
                    name='upsampling_logits')(logits)
    model = Model(inputs=inputs, outputs=logits)
    return model
