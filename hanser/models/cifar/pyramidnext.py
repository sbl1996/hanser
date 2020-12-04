from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Flatten

from hanser.models.layers import Pool2d, Conv2d, Norm, Act, GlobalAvgPool, Linear
from hanser.models.modules import PadChannel, DropPath, SELayer

__all__ = [
    "PyramidNeXt"
]


class AuxiliaryHead(Sequential):
    def __init__(self, in_channels, num_classes):
        layers = [
            Norm(in_channels),
            Pool2d(5, 3, padding=0, type='avg'),
            Conv2d(in_channels, 128, 1, norm='def', act='def'),
            Conv2d(128, 768, 2, padding=0, norm='def', act='def'),
            Flatten(),
            Linear(768, num_classes),
        ]
        super().__init__(layers)


class Shortcut(Sequential):
    def __init__(self, in_channels, out_channels, stride):
        layers = []
        if stride == 2:
            layers.append(Pool2d(2, 2, type='avg'))
        if in_channels != out_channels:
            layers.append((PadChannel(out_channels - in_channels)))
        super().__init__(layers)


class BasicBlock(Layer):

    def __init__(self, in_channels, channels, groups, stride=1, se_reduction=4, drop_path=0.2):
        super().__init__()
        assert groups == 1
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=3, norm='default', act='default'),
            *([Pool2d(3, 2)] if stride != 1 else []),
            Conv2d(channels, channels, kernel_size=3, norm='default'),
            *([SELayer(channels, se_reduction, groups)] if se_reduction else []),
            *([DropPath(drop_path)] if drop_path and stride == 1 else []),
        ]
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


class Bottleneck(Layer):

    def __init__(self, in_channels, channels, groups, stride=1, se_reduction=4, drop_path=0.2):
        super().__init__()
        branch1 = [
            Norm(in_channels),
            Conv2d(in_channels, channels, kernel_size=1, norm='default', act='default'),
            *([Pool2d(3, 2)] if stride != 1 else []),
            Conv2d(channels, channels, kernel_size=3, groups=groups, norm='default', act='default'),
            *([SELayer(channels, se_reduction, groups)] if se_reduction else []),
            Conv2d(channels, channels, kernel_size=1, norm='default'),
            *([DropPath(drop_path)] if drop_path and stride == 1 else []),
        ]
        self.branch1 = Sequential(branch1)
        self.branch2 = Shortcut(in_channels, channels, stride)

    def call(self, x):
        return self.branch1(x) + self.branch2(x)


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


class PyramidNeXt(Model):
    def __init__(self, start_channels, widening_fractor, depth, groups, se_reduction, drop_path, use_aux_head,
                 block='bottleneck', num_classes=10):
        super().__init__()

        if block == 'basic':
            num_layers = [(depth - 2) // 6] * 3
            block = BasicBlock
        elif block == 'bottleneck':
            num_layers = [(depth - 2) // 9] * 3
            block = Bottleneck
        else:
            raise ValueError("block must be `basic` or `bottleneck`, got %s" % block)

        self.num_layers = num_layers
        self.use_aux_head = use_aux_head

        strides = [1, 2, 2]

        add_channel = widening_fractor / sum(num_layers)
        in_channels = start_channels

        self.init_block = Conv2d(3, start_channels, kernel_size=3, norm='default')

        channels = start_channels
        k = 1
        units = []
        for i, (n, s) in enumerate(zip(num_layers, strides)):
            channels += add_channel
            units.append(block(in_channels, round_channels(channels, groups),
                               groups, stride=s, se_reduction=se_reduction, drop_path=drop_path))
            in_channels = round_channels(channels, groups)
            k += 1

            if i == 2 and self.use_aux_head:
                self.aux_head_index = k - 1
                self.aux_head = AuxiliaryHead(in_channels, num_classes)

            for j in range(1, n):
                channels = channels + add_channel
                units.append(block(in_channels, round_channels(channels, groups),
                                        groups, se_reduction=se_reduction, drop_path=drop_path))
                in_channels = round_channels(channels, groups)
                k += 1

        self.units = units
        self.post_activ = Sequential([
            Norm(in_channels),
            Act(),
        ])

        assert (start_channels + widening_fractor) == in_channels

        self.final_pool = GlobalAvgPool()
        self.fc = Linear(in_channels, num_classes)

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

def test_net():
    model = PyramidNeXt(16, 100-16, 110, 1, False, 0.0, False, block='basic')
    model.build((None, 32, 32, 3))
    model.summary()