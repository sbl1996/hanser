from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer

from hanser.models.layers import GlobalAvgPool, Conv2d


class SELayer(Layer):

    def __init__(self, in_channels, reduction, **kwargs):
        super().__init__(**kwargs)
        channels = min(max(in_channels // reduction, 32), in_channels)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.fc = Sequential([
            Conv2d(in_channels, channels, 1, act='def'),
            Conv2d(channels, in_channels, 1, act='sigmoid'),
        ])

    def call(self, x):
        s = self.pool(x)
        s = self.fc(s)
        return x * s