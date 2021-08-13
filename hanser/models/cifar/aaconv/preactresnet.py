from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.layers import Act, Conv2d, Norm, GlobalAvgPool, Linear, Identity, Dropout
from hanser.models.cifar.aaconv.layers import AAConv


class PreActResBlock(Layer):
    def __init__(self, in_channels, out_channels, stride, dropout=0,
                 n_heads=8, k=0.2, v=0.1):
        super().__init__()
        dk = int(k * out_channels)
        dv = int(v * out_channels)
        self.stride = stride
        self.norm1 = Norm(in_channels)
        self.act1 = Act()
        self.conv1 = AAConv(in_channels, out_channels, kernel_size=3, stride=stride,
                            n_heads=n_heads, dk=dk, dv=dv)
        self.norm2 = Norm(out_channels)
        self.act2 = Act()
        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if stride > 1 or in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.act1(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + shortcut


class ResNet(Model):

    def __init__(self, depth, widen_factor, dropout=0, num_classes=100,
                 stages=(16, 16, 32, 64), n_heads=8, k=0.2, v=0.1):
        super().__init__()
        num_blocks = (depth - 4) // 6
        stages = [stages[0], *[c * widen_factor for c in stages[1:]]]
        strides = [1, 2, 2]

        self.stem = Conv2d(3, stages[0], kernel_size=3)

        for i in range(3):
            layer = self._make_layer(
                stages[i], stages[i+1], num_blocks, stride=strides[i],
                dropout=dropout, n_heads=n_heads, k=k, v=v)
            setattr(self, "layer" + str(i+1), layer)

        self.norm = Norm(stages[3])
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, **kwargs):
        layers = [PreActResBlock(in_channels, out_channels, stride=stride, **kwargs)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, stride=1, **kwargs))
        return Sequential(layers)

    def call(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x