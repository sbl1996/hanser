import tensorflow as tf

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dropout

from hanser.models.layers import Conv2d, GlobalAvgPool, Linear
from hanser.models.modules import SELayer


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

    def __init__(self, in_channels, use_se, name):
        super().__init__(name=name)
        c = in_channels // 2
        branch2 = [
            Conv2d(c, c, kernel_size=1, norm='def', act='def', name='conv1'),
            Conv2d(c, c, kernel_size=3, groups=c, norm='def', name='conv2'),
            Conv2d(c, c, kernel_size=1, norm='def', act='def', name='conv3')
        ]
        if use_se:
            branch2.append(SELayer(c, reduction=2, name='se'))
        self.branch2 = Sequential(branch2, name='branch2')

    def call(self, x):
        c = x.shape[-1] // 2
        x1, x2 = tf.split(x, num_or_size_splits=[c, c], axis=-1)
        x2 = self.branch2(x2)
        x = tf.concat([x1, x2], axis=-1)
        return channel_shuffle(x, 2)


class ReduceCell(Layer):

    def __init__(self, in_channels, out_channels, use_se, name):
        super().__init__(name=name)
        c = out_channels // 2
        self.branch1 = Sequential([
            Conv2d(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels, norm='def', name='conv1'),
            Conv2d(in_channels, c, kernel_size=1, norm='def', act='def', name='conv2'),
        ], name='branch1')
        branch2 = [
            Conv2d(in_channels, c, kernel_size=1, norm='def', act='def', name='conv1'),
            Conv2d(c, c, kernel_size=3, stride=2, groups=c, norm='def', name='conv2'),
            Conv2d(c, c, kernel_size=1, norm='def', act='def', name='conv3')
        ]
        if use_se:
            branch2.append(SELayer(c, reduction=2, name='se'))
        self.branch2 = Sequential(branch2, name='branch2')

    def call(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = tf.concat([x1, x2], axis=-1)
        return channel_shuffle(x, 2)


def _make_layer(in_channels, out_channels, num_units, stride, use_se, name):
    layers = []
    if stride == 2:
        layers.append(ReduceCell(in_channels, out_channels, use_se, name='unit1'))
    else:
        layers.append(Conv2d(in_channels, out_channels, 3, norm='def', act='def', name='unit1'))
    for i in range(1, num_units):
        layers.append(NormalCell(out_channels, use_se, name=f'unit{i+1}'))
    return Sequential(layers, name=name)


class ShuffleNetV2(Model):

    def __init__(self, stem_channels=32, channels_per_stage=(128, 256, 512), units_per_stage=(4, 8, 4),
                 final_channels=1024, use_se=True, dropout=0.2, num_classes=10):
        super().__init__()
        self.dropout = dropout

        cs = [stem_channels] + list(channels_per_stage)
        ns = units_per_stage

        self.stem = Conv2d(3, stem_channels, 3, norm='def', act='def', name='stem')

        self.stage1 = _make_layer(cs[0], cs[1], ns[0], stride=1, use_se=use_se, name='stage1')
        self.stage2 = _make_layer(cs[1], cs[2], ns[1], stride=2, use_se=use_se, name='stage2')
        self.stage3 = _make_layer(cs[2], cs[3], ns[2], stride=2, use_se=use_se, name='stage3')

        self.final_conv = Conv2d(cs[-1], final_channels, 1, norm='def', act='def', name='final_conv')
        self.global_pool = GlobalAvgPool(name='global_pool')
        if dropout:
            self.dropout = Dropout(dropout, name='dropout')
        self.classifier = Linear(final_channels, num_classes, name='classifier')

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
