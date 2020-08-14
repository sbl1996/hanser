import math
import tensorflow as tf

import torch
import torch.nn as nn

from hanser.models.layers import Linear, set_defaults, Conv2d


# mt = nn.Conv2d(128, 512, 3)
# m = Conv2d(128, 512, 3)
# m.build((None, 32, 32, 128))
#
# print(mt.weight.std())
# print(m.layers[1].kernel.numpy().std())
#
# print(mt.bias.std())
# print(m.layers[1].bias.numpy().std())

# mt = nn.Linear(1024, 1024)
# m = Linear(1024, 1024)
# m.build((None, 1024))
#
# print(mt.weight.std())
# print(m.kernel.numpy().std())
#
# print(mt.bias.std())
# print(m.bias.numpy().std())
