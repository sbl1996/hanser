from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Conv2D, BatchNormalization, Concatenate, DepthwiseConv2D, Input, \
    GlobalAvgPool2D, Dense, Add


def ReLUConvBN(x, C_out, kernel_size, stride):
    x = Activation('relu')(x)
    x = Conv2D(C_out, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def FactorizedReduce(x, C_out):
    assert C_out % 2 == 0
    x = Activation('relu')(x)
    x1 = Conv2D(C_out // 2, 1, strides=2, use_bias=False)(x)
    x2 = Conv2D(C_out // 2, 1, strides=2, use_bias=False)(x[:, 1:, 1:, :])
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    return x


def SepConv(x, C_out, kernel_size, stride):
    C_in = x.shape[-1]
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = Conv2D(C_in, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = DepthwiseConv2D(kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = Conv2D(C_out, 1, use_bias=False)(x)
    x = BatchNormalization()(x)
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
    x = Conv2D(C_curr, 3, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

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
