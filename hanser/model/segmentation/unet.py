from tensorflow.python.keras.layers import ReLU, MaxPool2D, Concatenate

from hanser.model.layers import bn, conv2d, deconv2d


def conv_block(x, channels):
    x = conv2d(x, channels, kernel_size=3)
    x = bn(x)
    x = ReLU()(x)
    x = conv2d(x, channels, kernel_size=3)
    x = bn(x)
    x = ReLU()(x)
    return x


def unet(x, num_classes, channels=64):
    c0 = conv_block(x, channels * 1)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c0)
    c1 = conv_block(x, channels * 2)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c1)
    c2 = conv_block(x, channels * 4)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c2)
    c3 = conv_block(x, channels * 8)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c3)
    x = conv_block(x, channels * 16)

    print(x.shape)
    t = deconv2d(x, channels * 8, kernel_size=2, stride=2)
    print(t.shape)
    x = Concatenate()([c3, t])
    x = conv_block(x, channels * 8)

    x = Concatenate()([c2, deconv2d(x, channels * 4, kernel_size=2, stride=2)])
    x = conv_block(x, channels * 4)

    x = Concatenate()([c1, deconv2d(x, channels * 2, kernel_size=2, stride=2)])
    x = conv_block(x, channels * 2)

    x = Concatenate()([c0, deconv2d(x, channels * 1, kernel_size=2, stride=2)])
    x = conv_block(x, channels * 1)

    logits = conv2d(x, num_classes, kernel_size=1, use_bias=True)

    return logits