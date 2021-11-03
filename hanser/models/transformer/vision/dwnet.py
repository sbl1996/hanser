import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Conv2d, Dropout, Identity, Norm, GlobalAvgPool, Linear
from hanser.models.modules import DropPath, SpaceToDepth


# class DynamicDWConv(nn.Module):
#     def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
#         super().__init__()
#         self.dim = dim
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.groups = groups
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
#         self.bn = nn.BatchNorm2d(dim // reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.bias = None
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
#         weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
#         x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding,
#                      groups=b * self.groups)
#         x = x.view(b, c, x.shape[-2], x.shape[-1])
#         return x


class MLP(Layer):

    def __init__(self, in_channels, expand=4., act='gelu', dropout=0.):
        super().__init__()
        channels = int(in_channels * expand)
        self.fc1 = Conv2d(in_channels, channels, kernel_size=1, act=act)
        self.dropout1 = Dropout(dropout)
        self.fc2 = Conv2d(channels, in_channels, kernel_size=1)
        self.dropout2 = Dropout(dropout)

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class DWBlock(Layer):

    def __init__(self, channels, kernel_size, dynamic=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size  # Wh, Ww
        self._dynamic = dynamic

        self.conv0 = Conv2d(channels, channels, 1, norm='def', act='def')

        # if dynamic:
        #     self.conv = DynamicDWConv(channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=channels)
        # else:
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
                 expand=4., dropout=0., drop_path=0., dynamic=False, act='gelu'):
        super().__init__()

        self.attn2conv = DWBlock(channels, kernel_size, dynamic)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.mlp = MLP(channels, expand, act=act, dropout=dropout)
        self.bn = Norm(channels, 'bn')

    def call(self, x):

        shortcut = x
        print(shortcut.shape)
        x = self.attn2conv(x)
        print(x.shape)
        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.mlp(self.bn(x))
        x = shortcut + self.drop_path(x)
        return x


class PatchMerging(Layer):

    def __init__(self, in_channels, out_channels, norm='ln'):
        super().__init__()
        self.space_to_depth = SpaceToDepth(2)
        self.norm = Norm(in_channels * 4, norm)
        self.reduction = Conv2d(in_channels * 4, out_channels, 1, bias=False)

    def call(self, x):
        x = self.space_to_depth(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(Layer):

    def __init__(self, channels, depth, kernel_size,
                 expand=4., dropout=0., drop_path=0., dynamic=False):

        super().__init__()
        self.channels = channels
        self.depth = depth

        self.blocks = Sequential([
            SpatialBlock(channels=channels,
                         kernel_size=kernel_size,
                         expand=expand,
                         dropout=dropout,
                         dynamic=dynamic,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

    def call(self, x):
        x = self.blocks(x)
        return x


class DWNet(Model):

    def __init__(self, patch_size=4, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), kernel_size=7, expand=4.,
                 drop_rate=0., drop_path_rate=0.2, norm='ln',
                 dynamic=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.stem = Conv2d(3, embed_dim, patch_size, stride=patch_size, padding=0, norm=norm)
        self.pos_drop = Dropout(drop_rate)

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

        for i in range(self.num_layers):
            channels = int(embed_dim * 2 ** i)
            layer = BasicLayer(channels=channels,
                               depth=depths[i],
                               kernel_size=kernel_size,
                               expand=expand,
                               dropout=drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               dynamic=dynamic)
            if i != 0:
                downsample = PatchMerging(channels // 2, channels, norm=norm)
                layer = Sequential([downsample, layer])
            setattr(self, "layer" + str(i+1), layer)

        self.norm = Norm(self.num_features, norm)
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.num_features, num_classes)

    def call(self, x):
        x = self.stem(x)
        x = self.pos_drop(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x