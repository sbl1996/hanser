import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Identity, Norm, GlobalAvgPool, Linear
from hanser.models.modules import DropPath, SpaceToDepth


class MLP(Layer):

    def __init__(self, in_channels, expand=4.):
        super().__init__()
        channels = int(in_channels * expand)
        self.norm0 = Norm(in_channels)
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1, norm='def', act='def')
        self.conv2 = Conv2d(channels, in_channels, kernel_size=1, norm='def')

    def call(self, x):
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DWBlock(Layer):

    def __init__(self, channels, kernel_size, dynamic=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size  # Wh, Ww
        self._dynamic = dynamic

        self.conv0 = Conv2d(channels, channels, 1, norm='def', act='def')

        self.conv1 = Conv2d(
            channels, channels, kernel_size, groups=channels, norm='def', act='def')

        self.conv2 = Conv2d(channels, channels, 1, norm='def')

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SpatialBlock(Layer):

    def __init__(self, channels, kernel_size=7,
                 expand=4., drop_path=0., dynamic=False):
        super().__init__()

        self.attn2conv = DWBlock(channels, kernel_size, dynamic)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.mlp = MLP(channels, expand)

    def call(self, x):

        shortcut = x
        x = self.attn2conv(x)
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x


class PatchMerging(Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.space_to_depth = SpaceToDepth(2)
        self.norm = Norm(in_channels * 4)
        self.reduction = Conv2d(in_channels * 4, out_channels, 1, bias=False)

    def call(self, x):
        x = self.space_to_depth(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class DWNet(Model):

    def __init__(self, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2), kernel_size=7, expand=4.,
                 drop_path_rate=0.2, dynamic=False):
        super().__init__()

        num_layers = len(depths)

        self.stem = Conv2d(3, embed_dim, 7, stride=4, norm='def')

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

        k = 0
        for i in range(num_layers):
            channels = int(embed_dim * 2 ** i)
            blocks = []
            if i != 0:
                downsample = PatchMerging(channels // 2, channels)
                blocks.append(downsample)
            for j in range(depths[i]):
                blocks.append(SpatialBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    expand=expand,
                    dynamic=dynamic,
                    drop_path=dpr[k]))
                k += 1
            layer = Sequential(blocks)
            setattr(self, "layer" + str(i+1), layer)

        channels = int(embed_dim * 2 ** (num_layers - 1))
        self.norm = Norm(channels)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(channels, num_classes)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x