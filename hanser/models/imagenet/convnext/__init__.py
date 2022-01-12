import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant

from hanser.models.layers import Conv2d, Identity, Norm, Linear, GlobalAvgPool
from hanser.models.modules import DropPath


class Block(Layer):

    def __init__(self, channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        exp_channels = channels * 4
        self.dwconv = Conv2d(channels, channels, kernel_size=7, groups=channels, norm='ln')
        self.pwconv1 = Conv2d(channels, exp_channels, kernel_size=1, act='gelu')
        self.pwconv2 = Conv2d(exp_channels, channels, kernel_size=1)
        self.gamma = self.add_weight(
            shape=(channels,), initializer=Constant(layer_scale_init_value), trainable=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def call(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x + self.drop_path(identity)
        return x


class ConvNeXt(Model):

    def __init__(self, layers=(3, 3, 9, 3), channels=(96, 192, 384, 768),
                 num_classes=1000, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.stem = Conv2d(3, channels[0], kernel_size=4, stride=4, norm='ln')

        stages = []
        dp_rates = [x.numpy() for x in tf.linspace(0.0, drop_path, sum(layers))]
        cur = 0
        for i in range(4):
            stage = []
            if i != 0:
                downsample = Sequential([
                    Norm(channels[i], 'ln'),
                    Conv2d(channels[i-1], channels[i], kernel_size=2, stride=2)
                ])
                stage.append(downsample)
            stage.extend([
                Block(channels[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(layers[i])
            ])
            stages.append(Sequential(stage))
            cur += layers[i]

        self.stages = Sequential(stages)
        self.avgpool = GlobalAvgPool()
        self.norm = Norm(channels[-1], 'ln')
        self.fc = Linear(channels[-1], num_classes)

    def call(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.fc(x)
        return x

def convnext_tiny(**kwargs):
    return ConvNeXt(layers=[3, 3, 9, 3], channels=[96, 192, 384, 768], **kwargs)

def convnext_small(**kwargs):
    return ConvNeXt(layers=[3, 3, 27, 3], channels=[96, 192, 384, 768], **kwargs)

def convnext_base(**kwargs):
    return ConvNeXt(layers=[3, 3, 27, 3], channels=[128, 256, 512, 1024], **kwargs)

def convnext_large(**kwargs):
    return ConvNeXt(layers=[3, 3, 27, 3], channels=[192, 384, 768, 1536], **kwargs)

def convnext_xlarge(**kwargs):
    return ConvNeXt(layers=[3, 3, 27, 3], channels=[256, 512, 1024, 2048], **kwargs)