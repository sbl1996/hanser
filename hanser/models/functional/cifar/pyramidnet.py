from tensorflow.keras import Model
from tensorflow.keras.layers import Multiply, Add, Input

from hanser.models.functional.layers import conv2d, norm, act, pool, dense
from hanser.models.modules import PadChannel, DropPath, GlobalAvgPool

__all__ = [
    "PyramidNet"
]


def round_channels(channels, divisor=8, min_depth=None):
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def shortcut(x, out_channels, stride, name):
    if stride == 2:
        x = pool(x, 2, 2, type='avg', name=name + "/pool")
    in_channels = x.shape[-1]
    if in_channels != out_channels:
        x = PadChannel(out_channels - in_channels, name=name + "/pad")(x)
    return x


def se(x, groups, reduction, name):
    in_channels = x.shape[-1]
    channels = min(max(in_channels // reduction, 32), in_channels)
    channels = round_channels(channels, groups)
    s = GlobalAvgPool(keep_dim=True, name=name + "/squeeze")(x)
    s = conv2d(s, channels, 1, groups=groups,
               norm='default', act='default', name=name + "/fc1")
    s = conv2d(s, in_channels, 1, groups=groups,
               act='sigmoid', name=name + "/fc2")
    x = Multiply(name=name + "/excite")([x, s])
    return x


def bottleneck(x, channels, groups, stride=1, use_se=True, drop_path=0.2, name=None):
    expansion = 1
    out_channels = channels * expansion

    branch1 = norm(x, name=name + "/branch1/bn0")
    branch1 = conv2d(branch1, channels, kernel_size=1, norm='default', act='default', name=name + "/branch1/conv1")
    if stride != 1:
        branch1 = pool(branch1, 3, 2, name=name + "/branch1/pool")
    branch1 = conv2d(branch1, channels, 3, groups=groups,
                     norm='default', act='default', name=name + "/branch1/conv2")
    if use_se:
        branch1 = se(branch1, groups, reduction=4, name=name + "/branch1/se")
    branch1 = conv2d(branch1, out_channels, kernel_size=1, norm='default', name=name + "/branch1/conv3")
    if drop_path and stride == 1:
        branch1 = DropPath(drop_path, name=name + "/branch1/drop")(branch1)

    branch2 = shortcut(x, out_channels, stride, name=name + "/branch2")

    return Add(name=name + "/add")([branch1, branch2])


def PyramidNet(input_shape, start_channels, widening_fractor, depth, groups, use_se, drop_path, num_classes=10):
    num_layers = [(depth - 2) // 9] * 3
    strides = [1, 2, 2]

    add_channel = widening_fractor / sum(num_layers)
    add_rate = drop_path / sum(num_layers)
    in_channels = start_channels
    channels = start_channels
    drop_rate = add_rate

    inputs = Input(input_shape)

    x = conv2d(inputs, start_channels, 3, norm='default', name='init_block')

    for i, (n, s) in enumerate(zip(num_layers, strides)):
        channels = channels + add_channel
        x = bottleneck(x, round_channels(channels, groups), groups, stride=s, use_se=use_se, drop_path=drop_rate,
                       name=f"stage{i + 1}/unit1")
        drop_rate += add_rate
        in_channels = round_channels(channels, groups)
        for j in range(1, n):
            channels = channels + add_channel
            x = bottleneck(x, round_channels(channels, groups), groups, use_se=use_se, drop_path=drop_rate,
                           name=f"stage{i + 1}/unit{j + 1}")
            drop_rate += add_rate
            in_channels = round_channels(channels, groups)

    x = norm(x, name='post_activ/bn')
    x = act(x, name='post_activ/act')

    assert (start_channels + widening_fractor) == in_channels
    x = GlobalAvgPool(name="final_pool")(x)
    x = dense(x, num_classes, name="fc")
    model = Model(inputs=inputs, outputs=x)
    return model
