import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dropout

from hanser.models.layers import Conv2d, GlobalAvgPool, Linear
from hanser.models.attention import SELayer

__all__ = [
    'ShuffleNetV2'
]


def channel_shuffle(x, groups):
    b, h, w, c = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = c // groups

    x = tf.reshape(x, [b, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [b, h, w, c])
    return x


class NormalCell(Layer):

    def __init__(self, in_channels, use_se):
        super().__init__()
        c = in_channels // 2
        branch2 = [
            Conv2d(c, c, kernel_size=1, norm='def', act='def'),
            Conv2d(c, c, kernel_size=3, groups=c, norm='def'),
            Conv2d(c, c, kernel_size=1, norm='def', act='def')
        ]
        if use_se:
            branch2.append(SELayer(c, reduction=2))
        self.branch2 = Sequential(branch2)

    def call(self, x):
        c = x.shape[-1] // 2
        x1, x2 = tf.split(x, num_or_size_splits=[c, c], axis=-1)
        x2 = self.branch2(x2)
        x = tf.concat([x1, x2], axis=-1)
        return channel_shuffle(x, 2)


class ReduceCell(Layer):

    def __init__(self, in_channels, out_channels, use_se):
        super().__init__()
        c = out_channels // 2
        self.branch1 = Sequential([
            Conv2d(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels, norm='def'),
            Conv2d(in_channels, c, kernel_size=1, norm='def', act='def'),
        ])
        branch2 = [
            Conv2d(in_channels, c, kernel_size=1, norm='def', act='def'),
            Conv2d(c, c, kernel_size=3, stride=2, groups=c, norm='def'),
            Conv2d(c, c, kernel_size=1, norm='def', act='def')
        ]
        if use_se:
            branch2.append(SELayer(c, reduction=2))
        self.branch2 = Sequential(branch2)

    def call(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = tf.concat([x1, x2], axis=-1)
        return channel_shuffle(x, 2)


def _make_layer(in_channels, out_channels, num_units, stride, use_se):
    layers = []
    if stride == 2:
        layers.append(ReduceCell(in_channels, out_channels, use_se))
    else:
        layers.append(Conv2d(in_channels, out_channels, 3, norm='def', act='def'))
    for i in range(1, num_units):
        layers.append(NormalCell(out_channels, use_se))
    return Sequential(layers)


class ShuffleNetV2(Model):

    def __init__(self, stem_channels=32, channels_per_stage=(128, 256, 512), units_per_stage=(4, 8, 4),
                 final_channels=1024, use_se=True, dropout=0.2, num_classes=10):
        super().__init__()
        self.dropout = dropout

        cs = [stem_channels] + list(channels_per_stage)
        ns = units_per_stage

        self.stem = Conv2d(3, stem_channels, 3, norm='def', act='def')

        self.stage1 = _make_layer(cs[0], cs[1], ns[0], stride=1, use_se=use_se)
        self.stage2 = _make_layer(cs[1], cs[2], ns[1], stride=2, use_se=use_se)
        self.stage3 = _make_layer(cs[2], cs[3], ns[2], stride=2, use_se=use_se)

        self.final_conv = Conv2d(cs[-1], final_channels, 1, norm='def', act='def')
        self.global_pool = GlobalAvgPool()
        if dropout:
            self.dropout = Dropout(dropout)
        self.classifier = Linear(final_channels, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.final_conv(x)

        x = self.global_pool(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x