import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

from hanser.losses import CrossEntropy
from hanser.train.optimizers import SGD

from hanser.models.nas.operations import ReLUConvBN
from hanser.models.cifar.nasnet import NASNet
from hanser.models.nas.genotypes import Genotype

from horch.models.nas.cifar.darts import DARTS
from horch.models.utils import summary

FTSO = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5],
)


def copy_conv(src, dst):
    if isinstance(src, tf.keras.Sequential):
        src = src.layers[1]
    with torch.no_grad():
        if "Depth" in type(src).__name__:
            weight = src.depthwise_kernel
            dst.weight.copy_(torch.from_numpy(np.transpose(weight.numpy(), [2, 3, 0, 1])))
        else:
            weight = src.kernel
            dst.weight.copy_(torch.from_numpy(np.transpose(weight.numpy(), [3, 2, 0, 1])))
        if dst.bias:
            dst.bias.copy_(torch.from_numpy(src.bias.numpy()))


def copy_bn(src, dst):
    with torch.no_grad():
        dst.weight.copy_(torch.from_numpy(src.gamma.numpy()))
        dst.bias.copy_(torch.from_numpy(src.beta.numpy()))
        dst.running_mean.copy_(torch.from_numpy(src.moving_mean.numpy()))
        dst.running_var.copy_(torch.from_numpy(src.moving_variance.numpy()))


def copy_linear(src, dst):
    with torch.no_grad():
        weight = src.kernel
        dst.weight.copy_(torch.from_numpy(np.transpose(weight.numpy(), [1, 0])))
        if dst.bias is not None:
            dst.bias.copy_(torch.from_numpy(src.bias.numpy()))


def copy_preprocess(src, dst):
    if isinstance(src, ReLUConvBN):
        copy_conv(src.layers[1], dst.op[1])
        copy_bn(src.layers[2], dst.op[2])
    else:
        copy_conv(src.conv1, dst.conv_1)
        copy_conv(src.conv2, dst.conv_2)
        copy_bn(src.norm, dst.bn)


def copy_sep_conv(src, dst):
    copy_conv(src.layers[1], dst.op[1])
    copy_conv(src.layers[2], dst.op[2])
    copy_bn(src.layers[3], dst.op[3])
    copy_conv(src.layers[5], dst.op[5])
    copy_conv(src.layers[6], dst.op[6])
    copy_bn(src.layers[7], dst.op[7])


weight_decay = 0
num_layers = 20
channels = 4
net1 = NASNet(channels, num_layers, False, 0, 10, FTSO)
net1.build((None, 32, 32, 3))
net2 = DARTS(channels, num_layers, False, 0, 10, FTSO)

optimizer1 = SGD(
    0.025, momentum=0.9, weight_decay=weight_decay)
optimizer2 = torch.optim.SGD(
    net2.parameters(), 0.025, momentum=0.9, weight_decay=weight_decay)

criterion1 = CrossEntropy()
criterion2 = nn.CrossEntropyLoss()
# net1.summary()
# summary(net2, (3, 32,32))

copy_conv(net1.stem.layers[0], net2.stem[0])
copy_bn(net1.stem.layers[1], net2.stem[1])

for i in range(num_layers):
    cell1 = net1.cells[i]
    cell2 = net2.cells[i]
    copy_preprocess(cell1.preprocess0, cell2.preprocess0)
    copy_preprocess(cell1.preprocess1, cell2.preprocess1)
    copy_sep_conv(cell1._ops[0][0], cell2._ops[0][0])
    copy_sep_conv(cell1._ops[0][1], cell2._ops[0][1])
    copy_sep_conv(cell1._ops[1][0], cell2._ops[1][0])
    copy_sep_conv(cell1._ops[1][1], cell2._ops[1][1])
    copy_sep_conv(cell1._ops[2][0], cell2._ops[2][0])
    copy_sep_conv(cell1._ops[2][1], cell2._ops[2][1])
    copy_sep_conv(cell1._ops[3][0], cell2._ops[3][0])
    copy_sep_conv(cell1._ops[3][1], cell2._ops[3][1])

copy_linear(net1.classifier.layers[1], net2.classifier[1])

for _ in range(1):
    x1 = tf.random.normal([2, 32, 32, 3])
    x2 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))
    y1 = tf.random.uniform((2,), 0, 10, dtype=tf.int32)
    y2 = torch.from_numpy(y1.numpy()).to(torch.int64)
    y1 = tf.one_hot(y1, 10)

    with tf.GradientTape() as tape:
        p1 = net1(x1, training=True)
        loss1 = criterion1(y1, p1)
        loss1 = tf.reduce_mean(loss1)
    grads = tape.gradient(loss1, net1.trainable_variables)
    optimizer1.apply_gradients(zip(grads, net1.trainable_variables))

    net2.train()
    net2.zero_grad()
    p2 = net2(x2)
    loss2 = criterion2(p2, y2)
    loss2.backward()
    optimizer2.step()

    d2 = p1.numpy() - p2.detach().numpy()
    print(d2.mean(), d2.std())

    w1 = net1.stem.layers[0].layers[1].kernel
    w1 = np.transpose(w1.numpy(), [3, 2, 0, 1])
    w2 = net2.stem[0].weight.detach().numpy()
    dw = w1 - w2
    print(dw.mean(), dw.std())