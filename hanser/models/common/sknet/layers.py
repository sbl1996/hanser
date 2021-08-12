import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from hanser.models.layers import Conv2d, GlobalAvgPool


class SKConv(Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 reduction=2, norm='def', act='def'):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels * 2, kernel_size, stride=stride,
                            norm=norm, act=act)

        d = max(int(out_channels / reduction), 32)
        self.pool = GlobalAvgPool(keep_dim=True)
        self.fc = Sequential([
            Conv2d(out_channels, d, 1, norm='def', act='def'),
            Conv2d(d, out_channels * 2, 1),
        ])

    def call(self, x):
        x = self.conv1(x)
        xs = tf.split(x, num_or_size_splits=2, axis=-1)
        u = tf.reduce_sum(xs, axis=0)
        s = self.pool(u)
        z = self.fc(s)
        z = tf.split(z, num_or_size_splits=2, axis=-1)
        attn = tf.nn.softmax(z, axis=0)
        v = tf.reduce_sum(xs * attn, axis=0)
        return v