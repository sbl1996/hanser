import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dropout

from hanser.models.layers import Conv2d, Pool2d, Linear, GlobalAvgPool, Act
from hanser.models.attention import SELayer
from hanser.models.modules import StochDepth


class NFBlock(Layer):

    def __init__(
        self, in_channels, out_channels, stride=1,
        beta=1.0, alpha=0.2, expansion=2.25, se_ratio=0.5,
        group_width=8, drop_path=0.0, activation=None):
        super().__init__()
        self.beta, self.alpha = beta, alpha
        self.stride = stride
        conv_op = Conv2d(scaled_ws=True)


        width = int(in_channels * expansion)
        if group_width is None:
            groups = 1
        else:
            groups = width // group_width
            # Round width up if you pick a bad group size
            width = groups * group_width

        self.act1 = Act(activation)
        self.conv1 = conv_op(in_channels, width, kernel_size=1)

        self.act2 = Act(activation)
        self.conv2 = conv_op(width, width, kernel_size=3, stride=stride, groups=groups)

        self.se = SELayer(width, se_channels=int(width * se_ratio), scale=2.0)

        self.act3 = Act(activation)
        self.conv3 = conv_op(width, out_channels, kernel_size=1)

        self.drop_path = StochDepth(drop_path) if drop_path else None

        if stride > 1 or in_channels != out_channels:
            shortcut = [conv_op(in_channels, out_channels, kernel_size=1)]
            if stride > 1:
                shortcut.append(Pool2d(2, 2, type='avg'))
            self.shortcut = Sequential(shortcut)
        else:
            self.shortcut = None

        self.gain = self.add_weight(
            name='gain', shape=(), dtype=tf.float32,
            trainable=True, initializer='zeros')

    def call(self, x):
        shortcut = x
        x = self.act1(x) * self.beta
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(self.act2(x))
        x = self.se(x)
        x = self.conv3(self.act3(x))
        if self.drop_path is not None:
            x = self.drop_path(x)
        return x * self.gain * self.alpha + shortcut

    def count_flops(self):
        from hanser.models.profile.base import count_conv
        h, w, c = self.input_shape[1:]
        conv1_flops = count_conv(self.conv1, h, w)
        conv2_flops = count_conv(self.conv2.layers[1], h, w)
        if self.stride > 1:
            h, w = h / self.stride, w / self.stride
        if self.shortcut is not None:
            sc_flops = count_conv(self.shortcut.layers[0], h, w)
        else:
            sc_flops = 0
        # SE flops happen on avg-pooled activations
        se_flops = count_conv(self.se.fc.layers[0].layers[0], 1, 1)
        se_flops += count_conv(self.se.fc.layers[1].layers[0], 1, 1)
        conv3_flops = count_conv(self.conv3, h, w)
        return sum([conv1_flops, conv2_flops, se_flops, conv3_flops, sc_flops])


class NFRegNet(Model):

    def __init__(self, widths=(48, 104, 208, 440), depths=(1, 3, 6, 6),
                 num_classes=1000, width_mul=0.75, expansion=2.25,
                 se_ratio=0.5, group_width=8, alpha=0.2, dropout=0.0,
                 activation='swish', drop_path=0.1):
        super().__init__()
        activation = 'scaled_' + activation

        conv_op = Conv2d(scaled_ws=True)

        widths = [int(val * width_mul) for val in widths]

        in_channels = widths[0]
        self.stem = conv_op(3, in_channels, kernel_size=3, stride=2)

        num_blocks = sum(depths)
        expected_std = 1.0
        index = 0
        for i, (block_width, stage_depth) in enumerate(zip(widths, depths)):
            layers = []
            for block_index in range(stage_depth):
                # Following EffNets, do not expand first block
                expand_ratio = 1 if index == 0 else expansion
                beta = 1. / expected_std
                drop_path_rate = drop_path * index / num_blocks
                block = NFBlock(
                    in_channels, block_width, stride=2 if block_index == 0 else 1,
                    beta=beta, alpha=alpha, expansion=expand_ratio, se_ratio=se_ratio,
                    group_width=group_width, drop_path=drop_path_rate, activation=activation)
                layers.append(block)

                in_channels = block_width
                if block_index == 0:
                    expected_std = 1.
                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5
                index += 1

            setattr(self, "layer%d" % (i + 1), Sequential(layers))

        ch = int(1280 * in_channels // 440)
        self.final_conv = conv_op(in_channels, ch, kernel_size=1, act=activation)
        in_channels = ch

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout > 0.0 else None
        self.fc = Linear(in_channels, num_classes, kernel_init='zeros')

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_conv(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_flops(self):
        from hanser.models.profile.base import count_conv, count_linear
        h, w, c = self.layers[0].input_shape[1:]
        flops = [count_conv(self.stem, h, w)]
        h, w = h / 2, w / 2

        for i in range(4):
            for block in getattr(self, "layer" + str(i + 1)).layers:
                flops.append(block.count_flops())
                if block.stride > 1:
                    h, w = h / block.stride, w / block.stride

        flops.append(count_conv(self.final_conv, h, w))
        flops.append(count_linear(self.fc))
        return flops, sum(flops)

# Block base widths and depths for each variant
params = {'NF-RegNet-B0': {'width': [48, 104, 208, 440], 'depth': [1, 3, 6, 6],
                           'train_imsize': 192, 'test_imsize': 224,
                           'weight_decay': 2e-5, 'drop_rate': 0.2},
          'NF-RegNet-B1': {'width': [48, 104, 208, 440], 'depth': [2, 4, 7, 7],
                           'train_imsize': 224, 'test_imsize': 256,
                           'weight_decay': 2e-5, 'drop_rate': 0.2},
          'NF-RegNet-B2': {'width': [56, 112, 232, 488], 'depth': [2, 4, 8, 8],
                           'train_imsize': 240, 'test_imsize': 272,
                           'weight_decay': 3e-5, 'drop_rate': 0.3},
          'NF-RegNet-B3': {'width': [56, 128, 248, 528], 'depth': [2, 5, 9, 9],
                           'train_imsize': 288, 'test_imsize': 320,
                           'weight_decay': 4e-5, 'drop_rate': 0.3},
          'NF-RegNet-B4': {'width': [64, 144, 288, 616], 'depth': [2, 6, 11, 11],
                           'train_imsize': 320, 'test_imsize': 384,
                           'weight_decay': 4e-5, 'drop_rate': 0.4},
          'NF-RegNet-B5': {'width': [80, 168, 336, 704], 'depth': [3, 7, 14, 14],
                           'train_imsize': 384, 'test_imsize': 456,
                           'weight_decay': 5e-5, 'drop_rate': 0.4}}


def _get_nf_regnet(name, **kwargs):
    config = params[name]
    return NFRegNet(
        widths=config['width'], depths=config['depth'], dropout=config['drop_rate'], **kwargs)


def NF_RegNet_B0(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B0', num_classes=num_classes, **kwargs)


def NF_RegNet_B1(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B1', num_classes=num_classes, **kwargs)


def NF_RegNet_B2(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B2', num_classes=num_classes, **kwargs)


def NF_RegNet_B3(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B3', num_classes=num_classes, **kwargs)


def NF_RegNet_B4(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B4', num_classes=num_classes, **kwargs)


def NF_RegNet_B5(num_classes=1000, **kwargs):
    return _get_nf_regnet('NF-RegNet-B5', num_classes=num_classes, **kwargs)
