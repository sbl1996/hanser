from tensorflow.keras.layers import Input
from hanser.models.cifar.pyramidnext import PyramidNeXt
from hanser.models.layers import DEFAULTS
DEFAULTS['weight_decay'] = 1e-4

input_shape = (32, 32, 3)
net = PyramidNeXt(4, 32-4, 20, 1, True, 0.2, 10)

input = Input(input_shape)
net.call(input)
net.build((None, *input_shape))

net.summary()


def get_children(m, name):
    try:
        l = getattr(m, name)
        return l
    except AttributeError as e:
        for l in m.layers:
            if l.name == name:
                return l
        raise e


def ch(m, *ks):
    for k in ks:
        m = get_children(m, k)
    m.summary()
    return m