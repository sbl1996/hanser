import tensorflow as tf
from tensorflow.keras import Sequential

from hanser.models.layers import Conv2d, Act, Norm, set_defaults

set_defaults({
    'weight_decay': 1e-4,
})

m = Sequential([
    Conv2d(3, 32, 3, norm='def'),
    Act(name='act1'),
    Conv2d(32, 32, 3, stride=1, bias=False, name='depthwise1'),
    Conv2d(32, 32, 1, bias=False, name='pointwise1'),
    Norm(32, name='norm1'),
])

m.build((None, 32, 32, 3))
len(m.losses)