import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Norm, GlobalAvgPool, Linear


class MLP(Layer):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.fc1 = Conv2d(in_channels, channels, 1, 1, act='gelu')
        self.fc2 = Conv2d(channels, in_channels, 1, 1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class PATM(Layer):
    def __init__(self, channels):
        super().__init__()

        self.fc_h = Conv2d(channels, channels, 1, 1, bias=False)
        self.fc_w = Conv2d(channels, channels, 1, 1, bias=False)
        self.fc_c = Conv2d(channels, channels, 1, 1, bias=False)

        self.theta_h_conv = Conv2d(channels, channels, 1, norm='bn', act='relu')
        self.theta_w_conv = Conv2d(channels, channels, 1, norm='bn', act='relu')

        self.tfc_h = Conv2d(2 * channels, channels, (7, 1), groups=channels, bias=False)
        self.tfc_w = Conv2d(2 * channels, channels, (1, 7), groups=channels, bias=False)

        self.proj = Conv2d(channels, channels, 1, 1)

    def call(self, x):

        x_h = self.fc_h(x)

        theta_h = self.theta_h_conv(x) # N, H, W, C
        x_h = tf.concat([x_h * tf.cos(theta_h), x_h * tf.sin(theta_h)], axis=-1)

        h = self.tfc_h(x_h)

        x_w = self.fc_w(x)
        theta_w = self.theta_w_conv(x)
        x_w = tf.concat([x_w * tf.cos(theta_w), x_w * tf.sin(theta_w)], axis=-1)
        w = self.tfc_w(x_w)

        c = self.fc_c(x)
        x = h + w + c
        x = self.proj(x)
        return x


class WaveBlock(Layer):

    def __init__(self, channels, mlp_ratio=4.):
        super().__init__()
        self.norm1 = Norm(channels)
        self.attn = PATM(channels)
        self.norm2 = Norm(channels)
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = MLP(channels, mlp_hidden_dim)

    def call(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WaveNet(Model):

    def __init__(self, channels=(64, 128, 320, 512), layers=(2, 2, 4, 2), num_classes=1000):
        super().__init__()
        self.stem = Conv2d(3, channels[0], kernel_size=7, stride=4, norm='bn')

        for i in range(len(layers)):
            blocks = []
            if i != 0:
                downsample = Conv2d(channels[i-1], channels[i], kernel_size=3, stride=2, norm='bn')
                blocks.append(downsample)
            for j in range(layers[i]):
                block = WaveBlock(channels[i])
                blocks.append(block)
            layer = Sequential(blocks)
            setattr(self, "layer{}".format(i+1), layer)

        self.norm = Norm(channels[-1])
        self.avg_pool = GlobalAvgPool()
        self.fc = Linear(channels[-1], num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


def WaveMLP_T(**kwargs):
    layers = [2, 2, 4, 2]
    channels = [64, 128, 320, 512]
    return WaveNet(channels, layers, **kwargs)