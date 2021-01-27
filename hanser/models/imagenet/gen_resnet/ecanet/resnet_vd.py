import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv1D

from hanser.models.layers import Conv2d, Act, Identity, GlobalAvgPool, Pool2d, Norm
from hanser.models.imagenet.gen_resnet.resnet_vd import ResNet as GenResNet


class ECALayer(Layer):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = GlobalAvgPool(keep_dim=True)
        self.conv = Conv1D(1, kernel_size=kernel_size, padding='SAME', use_bias=False)

    def call(self, x):
        y = self.avg_pool(x)

        y = tf.transpose(tf.squeeze(y, axis=1), [0, 2, 1])
        y = self.conv(y)
        y = tf.expand_dims(tf.transpose(y, [0, 2, 1]), 1)

        y = tf.sigmoid(y)
        return x * y


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, zero_init_residual=True, eca_kernel_size=3):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1,
                            norm='def', act='def')
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride,
                            norm='def', act='def')
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)
        self.bn3 = Norm(out_channels, gamma_init='zeros' if zero_init_residual else 'ones')
        self.eca = ECALayer(kernel_size=eca_kernel_size)

        if stride != 1 or in_channels != out_channels:
            shortcut = []
            if stride != 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            shortcut.append(
                Conv2d(in_channels, out_channels, kernel_size=1, norm='def'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = Identity()

        self.act = Act()

    def call(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.eca(x)
        x = x + identity
        x = self.act(x)
        return x


class ResNet(GenResNet):

    def __init__(self, block, layers, num_classes=1000, stages=(64, 64, 128, 256, 512),
                 zero_init_residual=False, eca_kernel_size=3):
        super().__init__(
            block, layers, num_classes, stages,
            zero_init_residual=zero_init_residual, eca_kernel_size=eca_kernel_size,
        )


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)