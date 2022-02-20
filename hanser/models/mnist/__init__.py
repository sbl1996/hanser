from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
import tensorflow as tf

from hanser.models.layers import Conv2d, Pool2d, Linear


class LeNet5(Sequential):

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__([
            Conv2d(in_channels, 6, kernel_size=5, norm='def', act='def'),
            Pool2d(2, 2, type='avg'),
            Conv2d(6, 16, kernel_size=5, norm='def', act='def'),
            Pool2d(2, 2, type='avg'),
            Flatten(),
            Linear(8 * 8 * 16, 120),
            Linear(120, 84),
            Linear(84, num_classes),
        ])