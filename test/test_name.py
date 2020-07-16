from tensorflow.keras.layers import Input
from hanser.models.cifar.pyramidnext import PyramidNeXt

input_shape = (32, 32, 3)
net = PyramidNeXt(4, 32-4, 20, 1, True, 0.2, 10)

input = Input(input_shape)
net.call(input)
net.build((None, *input_shape))

net.summary()

