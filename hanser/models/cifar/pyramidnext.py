from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel, DropPath, SELayer

__all__ = [
    "PyramidNeXt"
]


class AuxiliaryHead(Sequential):
    def __init__(self, in_channels, num_classes, name):
        layers = [
            Norm(in_channels, name='norm0'),
            Pool2d(5, 3, padding='valid', type='avg', name='pool'),
            Conv2d(in_channels, 128, 1, norm='def', act='def', name='conv1'),
            Conv2d(128, 768, 2, padding='valid', norm='def', act='def', name='conv2'),
            Flatten(name='flatten'),
            Linear(768, num_classes, name='fc'),
        ]
        super().__init__(layers, name=name)


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride, name):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg', name="pool"))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels, name="pad")))
        super().__init__(layers, name=name)


class Bottleneck(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, groups, stride=1, use_se=True, drop_path=0.2, name=None):
        super().__init__(name=name)
        out_channels = channels * self.expansion
        branch1 = [
            Norm(in_channels, name="norm0"),
            Conv2d(in_channels, channels, kernel_size=1, norm='default', act='default', name="conv1"),
            *([Pool2d(3, 2, name="pool")] if stride != 1 else []),
            Conv2d(channels, channels, kernel_size=3, groups=groups, norm='default', act='default', name="conv2"),
            *([SELayer(channels, 4, groups, name="se")] if use_se else []),
            Conv2d(channels, out_channels, kernel_size=1, norm='default', name="conv3"),
            *([DropPath(drop_path, name="drop")] if drop_path and stride == 1 else []),
        ]
        self.branch1 = Sequential(branch1, name="branch1")
        self.branch2 = Shortcut(in_channels, out_channels, stride, name="branch2")

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class PyramidNeXt(Model):
    def __init__(self, start_channels, widening_fractor, depth, groups, use_se, drop_path, use_aux_head,
                 num_classes=10):
        super().__init__()

        num_layers = [(depth - 2) // 9] * 3
        self.num_layers = num_layers
        self.use_aux_head = use_aux_head

        strides = [1, 2, 2]

        add_channel = widening_fractor / sum(num_layers)
        in_channels = start_channels

        self.init_block = Conv2d(3, start_channels, kernel_size=3, norm='default', name="init_block")

        channels = start_channels
        k = 1
        units = []
        for i, (n, s) in enumerate(zip(num_layers, strides)):
            channels += add_channel
            units.append(Bottleneck(in_channels, round_channels(channels, groups),
                                    groups, stride=s, use_se=use_se, drop_path=drop_path, name=f"unit{k}"))
            in_channels = round_channels(channels, groups) * Bottleneck.expansion
            k += 1

            if i == 2 and self.use_aux_head:
                self.aux_head_index = k - 1
                self.aux_head = AuxiliaryHead(in_channels, num_classes, name='aux_head')

            for j in range(1, n):
                channels = channels + add_channel
                units.append(Bottleneck(in_channels, round_channels(channels, groups),
                                        groups, use_se=use_se, drop_path=drop_path, name=f"unit{k}"))
                in_channels = round_channels(channels, groups) * Bottleneck.expansion
                k += 1

        self.units = units
        self.post_activ = Sequential([
            Norm(in_channels, name="norm"),
            Act(name="act"),
        ], name="post_activ")

        assert (start_channels + widening_fractor) * Bottleneck.expansion == in_channels

        self.final_pool = GlobalAvgPool(name="final_pool")
        self.fc = Linear(in_channels, num_classes, name="fc")

    def call(self, x):
        x = self.init_block(x)
        for i, unit in enumerate(self.units):
            x = unit(x)
            if self.use_aux_head and i == self.aux_head_index - 1:
                logits_aux = self.aux_head(x)
        x = self.post_activ(x)

        x = self.final_pool(x)
        x = self.fc(x)
        if self.use_aux_head:
            return x, logits_aux
        else:
            return x
