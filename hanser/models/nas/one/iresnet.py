import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import Constant

from hanser.models.modules import DropPath
from hanser.models.layers import GlobalAvgPool, Linear

from hanser.train.callbacks import Callback


class DropPathRateSchedule(Callback):

    def __init__(self, drop_path):
        super().__init__()
        self.drop_path = drop_path

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        rate = (epoch + 1) / epochs * self.drop_path

        self.learner.model.drop_rate.assign(rate)

        for l in self.learner.model.submodules:
            if isinstance(l, DropPath):
                l.rate.assign(rate)


class LayerArgs:

    def __init__(self, args):
        self.args = args


def _get_layer_kwargs(kwargs, i):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, LayerArgs):
            d[k] = v.args[i]
        else:
            d[k] = v
    return d


def _make_layer(block, in_channels, channels, blocks, stride, return_seq=True, **kwargs):
    layers = [block(in_channels, channels, stride=stride,
                    start_block=True, **_get_layer_kwargs(kwargs, 0))]
    in_channels = channels * block.expansion
    for i in range(1, blocks - 1):
        layers.append(block(in_channels, channels, stride=1,
                            exclude_bn0=i == 1, **_get_layer_kwargs(kwargs, i)))
    layers.append(block(in_channels, channels, stride=1,
                        end_block=True, **_get_layer_kwargs(kwargs, blocks - 1)))
    if return_seq:
        layers = Sequential(layers)
    return layers


def _get_kwargs(kwargs, i, layers=None, n=4):
    d = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            if len(v) == n:
                d[k] = v[i]
            elif layers is not None and len(v) == sum(layers):
                d[k] = LayerArgs(v[sum(layers[:i]):sum(layers[:(i+1)])])
            else:
                d[k] = v
        else:
            d[k] = v
    return d


class _IResNet(Model):

    def __init__(self, stem, block, depths, num_classes, channels,
                 strides, drop_path=0.5, dropout=0, **kwargs):
        super().__init__()

        assert len(depths) == len(channels) == len(strides)

        self.n_stages = len(strides)

        self.drop_rate = self.add_weight(
            name="drop_rate", shape=(), dtype=tf.float32,
            initializer=Constant(drop_path), trainable=False)

        drop_paths = []
        for start, end in depths:
            for i in range(start):
                drop_paths.append(0.)
            for i in range(start, end):
                drop_paths.append(drop_path)
        layers = [d[1] for d in depths]

        kwargs = {**kwargs, "drop_path": tuple(drop_paths)}

        self.stem = stem
        c_in = stem.out_channels

        for i, (c, n, s) in enumerate(zip(channels, layers, strides)):
            layer = _make_layer(
                block, c_in, c, n, s, **_get_kwargs(kwargs, i, layers, self.n_stages))
            c_in = c * block.expansion
            setattr(self, "layer" + str(i+1), layer)

        self.avgpool = GlobalAvgPool()
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Linear(c_in, num_classes)

    def call(self, x):
        x = self.stem(x)

        for i in range(self.n_stages):
            layer = getattr(self, "layer" + str(i+1))
            x = layer(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def l2_loss(self):
        weights1 = []
        weights2 = []
        weights1.extend(self.stem.trainable_variables)
        weights1.extend(self.fc.trainable_variables)

        for i in range(self.n_stages):
            layer = getattr(self, "layer" + str(i + 1))
            for l in layer.layers:
                if isinstance(l.drop_path, DropPath):
                    weights2.extend(l.trainable_variables)
                else:
                    weights1.extend(l.trainable_variables)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights1])
        if weights2:
            decay_rate = 1.0 - tf.cast(self.drop_rate, tf.float32)
            l2_loss = l2_loss + decay_rate * tf.add_n([tf.nn.l2_loss(v) for v in weights2])
        return l2_loss
