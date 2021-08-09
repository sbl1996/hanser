from tensorflow.keras.layers import Layer, Dropout

from hanser.models.layers import NormAct, Conv2d, Identity


class BasicBlock(Layer):
    expansion = 1

    def __init__(self, in_channels, channels, stride, dropout):
        super().__init__()
        out_channels = channels * self.expansion

        self.norm_act1 = NormAct(in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.norm_act2 = NormAct(out_channels)
        self.dropout = Dropout(dropout) if dropout else Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def call(self, x):
        shortcut = x
        x = self.norm_act1(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm_act2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + shortcut


class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_channels, channels, stride, dropout=0):
        super().__init__()
        out_channels = channels * self.expansion

        self.norm_act1 = NormAct(in_channels)
        self.conv1 = Conv2d(in_channels, channels, kernel_size=1)

        self.norm_act2 = NormAct(channels)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=stride)

        self.norm_act3 = NormAct(channels)
        self.conv3 = Conv2d(channels, out_channels, kernel_size=1)

        if in_channels != out_channels or stride != 1:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = None

    def call(self, x):
        shortcut = x
        x = self.norm_act1(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm_act2(x)
        x = self.conv2(x)
        x = self.norm_act3(x)
        x = self.conv3(x)
        return x + shortcut