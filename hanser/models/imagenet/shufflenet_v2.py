import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Pool2d, Linear, GlobalAvgPool

__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]


def channel_shuffle(x, groups):
    b, h, w, c = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = c // groups

    x = tf.reshape(x, [b, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [b, h, w, c])
    return x



class InvertedResidual(Layer):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride

        channels = out_channels // 2
        assert (self.stride != 1) or (in_channels == channels * 2)

        if self.stride > 1:
            self.branch1 = Sequential([
                Conv2d(in_channels, in_channels, kernel_size=3, stride=self.stride, groups=in_channels,
                       norm='def'),
                Conv2d(in_channels, channels, kernel_size=1, norm='def', act='def'),
            ])
        else:
            self.branch1 = Identity()

        self.branch2 = Sequential([
            Conv2d(in_channels if (self.stride > 1) else channels, channels, kernel_size=1,
                   norm='def', act='def'),
            Conv2d(channels, channels, kernel_size=3, stride=self.stride, groups=channels,
                   norm='def'),
            Conv2d(channels, channels, kernel_size=1, norm='def', act='def'),
        ])

    def call(self, x):
        if self.stride == 1:
            c = x.shape[-1] // 2
            x1, x2 = tf.split(x, num_or_size_splits=[c, c], axis=-1)
            out = tf.concat((x1, self.branch2(x2)), axis=-1)
        else:
            out = tf.concat((self.branch1(x), self.branch2(x)), axis=-1)

        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(Model):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        in_channels = 3
        out_channels = self._stage_out_channels[0]
        self.stem = Sequential([
            Conv2d(in_channels, out_channels, 3, stride=2, norm='def', act='def'),
            Pool2d(3, stride=2, type='max'),
        ])
        in_channels = out_channels

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, out_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(in_channels, out_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(out_channels, out_channels, 1))
            setattr(self, name, Sequential(seq))
            in_channels = out_channels

        out_channels = self._stage_out_channels[-1]
        self.final_conv = Conv2d(in_channels, out_channels, kernel_size=1,
                            norm='def', act='def')

        self.avgpool = GlobalAvgPool()
        self.fc = Linear(out_channels, num_classes)

    def call(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def shufflenet_v2_x0_5(**kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(**kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(**kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(**kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
