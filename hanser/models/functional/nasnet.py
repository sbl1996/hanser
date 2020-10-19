from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Concatenate, DepthwiseConv2D, Input, \
    GlobalAvgPool2D, Dense, Add


def relu(x):
    return Activation('relu')(x)


def bn(x):
    return BatchNormalization(momentum=0.9, epsilon=1e-5)(x)


def conv2d(x, channels, kernel_size, stride=1, groups=1, bias=False):
    if groups != 1:
        return DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=bias,
                               depthwise_initializer='he_uniformV2')(x)
    return Conv2D(channels, kernel_size, strides=stride, padding='same', use_bias=bias,
                  kernel_initializer='he_uniformV2')(x)


def ReLUConvBN(x, C_out, kernel_size, stride):
    x = relu(x)
    x = conv2d(x, C_out, kernel_size, stride=stride)
    x = bn(x)
    return x


def FactorizedReduce(x, C_out):
    assert C_out % 2 == 0
    x = relu(x)
    x1 = conv2d(x, C_out // 2, 1, stride=2)
    x2 = conv2d(x[:, 1:, 1:, :], C_out // 2, 1, stride=2)
    x = Concatenate()([x1, x2])
    x = bn(x)
    return x


def SepConv(x, C_out, kernel_size, stride):
    C_in = x.shape[-1]
    x = relu(x)
    x = conv2d(x, C_in, kernel_size, stride=stride, groups=C_in)
    x = conv2d(x, C_in, 1)
    x = bn(x)
    x = relu(x)
    x = conv2d(x, C_in, kernel_size, stride=1, groups=C_in)
    x = conv2d(x, C_out, 1)
    x = bn(x)
    return x


def Cell(s0, s1, C, reduction, reduction_prev):
    if reduction_prev:
        s0 = FactorizedReduce(s0, C)
    else:
        s0 = ReLUConvBN(s0, C, 1, 1)
    s1 = ReLUConvBN(s1, C, 1, 1)

    stride = 2 if reduction else 1
    s2 = Add()([SepConv(s0, C, 3, stride), SepConv(s1, C, 3, stride)])
    s3 = Add()([SepConv(s0, C, 3, stride), SepConv(s1, C, 3, stride)])
    s4 = Add()([SepConv(s0, C, 3, stride), SepConv(s1, C, 3, stride)])
    s5 = Add()([SepConv(s0, C, 3, stride), SepConv(s1, C, 3, stride)])

    return Concatenate()([s2, s3, s4, s5])


def NASNet(input_size, C, layers, num_classes):
    inputs = Input(input_size)

    stem_multiplier = 3
    C_curr = stem_multiplier * C
    x = conv2d(inputs, C_curr, 3)
    x = bn(x)

    s0 = s1 = x

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    reduction_prev = False
    for i in range(layers):
        if i in [layers // 3, 2 * layers // 3]:
            C_curr *= 2
            reduction = True
        else:
            reduction = False
        s0, s1 = s1, Cell(s0, s1, C_curr, reduction, reduction_prev)
        reduction_prev = reduction
        C_prev_prev, C_prev = C_prev, 4 * C_curr

    x = GlobalAvgPool2D()(s1)
    logits = Dense(num_classes)(x)

    model = Model(inputs=inputs, outputs=logits)
    return model
