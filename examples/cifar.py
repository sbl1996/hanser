import tensorflow as tf
from hanser.models.cifar.pyramidnet2 import PyramidNet

net = PyramidNet(16, 16, 11, 1, 10)

net.build_graph((None, 32, 32, 3))
net.summary()
x = tf.random.normal([1, 32, 32, 3])
y = net(x)