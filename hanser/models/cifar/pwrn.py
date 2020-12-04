from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer

from hanser.models.modules import Dropout
from hanser.models.layers import Act, Conv2d, Norm, GlobalAvgPool, Linear, Identity


class PreActResBlock(Sequential):
    def __init__(self, in_channels, out_channels, dropout):
        layers = [
            Norm(in_channels),
            Conv2d(in_channels, out_channels, kernel_size=3),
            Norm(out_channels),
            Act(),
            Conv2d(out_channels, out_channels, kernel_size=3),
            Norm(in_channels),
        ]
        if dropout:
            layers.insert(5, Dropout(dropout))
        super().__init__(layers)

    def call(self, x, training=None):
        return x + super().call(x, training)


class ResNet(Model):
    stages = [16, 16, 32, 64]

    def __init__(self, depth, k, dropout=0, avg_down=False, num_classes=10):
        super().__init__()
        num_blocks = (depth - 4) // 6
        self.conv = Conv2d(3, self.stages[0], kernel_size=3, norm='def')

        self.layer1 = self._make_layer(
            self.stages[0] * 1, self.stages[1] * k, num_blocks, stride=1,
            dropout=dropout, avg_down=avg_down)
        self.layer2 = self._make_layer(
            self.stages[1] * k, self.stages[2] * k, num_blocks, stride=2,
            dropout=dropout, avg_down=avg_down)
        self.layer3 = self._make_layer(
            self.stages[2] * k, self.stages[3] * k, num_blocks, stride=2,
            dropout=dropout, avg_down=avg_down)

        self.norm = Norm(self.stages[3] * k)
        self.act = Act()
        self.avgpool = GlobalAvgPool()
        self.fc = Linear(self.stages[3] * k, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [PreActDownBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for i in range(1, blocks):
            layers.append(
                PreActResBlock(out_channels, out_channels, dropout=dropout))
        return Sequential(layers)

    def call(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.norm(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x