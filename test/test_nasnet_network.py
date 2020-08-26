import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = x.new_empty((x.size(0), 1, 1, 1)).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


CDARTS = Genotype(
    normal=[
        ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1), ('sep_conv_3x3', 2),
        ('sep_conv_5x5', 0), ('sep_conv_3x3', 2),
        ('skip_connect', 0), ('sep_conv_3x3', 1),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0), ('sep_conv_5x5', 1),
        ('skip_connect', 0), ('skip_connect', 2),
        ('avg_pool_3x3', 0), ('dil_conv_5x5', 3),
        ('dil_conv_3x3', 0), ('dil_conv_3x3', 1)
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
    if isinstance(dst, ReLUConvBN):
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

def copy_dil_conv(src, dst):
    copy_conv(src.layers[1], dst.op[1])
    copy_conv(src.layers[2], dst.op[2])
    copy_bn(src.layers[3], dst.op[3])


from hanser.models.cifar.nasnet import NASNet
from hanser.models.layers import set_defaults
from hanser.losses import CrossEntropy

weight_decay = 0
set_defaults({
    "bn": {
        "fused": True,
        "eps": 1e-5,
    },
    "weight_decay": weight_decay,
})
num_layers = 8
net1 = NASNet(16, num_layers, False, 0, 10, CDARTS)
net1.build((None, 32, 32, 3))
net2 = NetworkCIFAR(16, 10, num_layers, False, CDARTS)
net2.drop_path_prob = 0

optimizer1 = tf.keras.optimizers.SGD(0.1, momentum=0.9, nesterov=False)
optimizer2 = torch.optim.SGD(
    net2.parameters(), 0.1, momentum=0.9, dampening=0.9, nesterov=False, weight_decay=weight_decay)

criterion1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
criterion2 = torch.nn.CrossEntropyLoss()
# from horch.models.utils import summary
# net1.summary()
# summary(net2)


copy_conv(net1.stem.layers[0], net2.stem[0])
copy_bn(net1.stem.layers[1], net2.stem[1])

for i in range(num_layers):
    cell1 = net1.cells[i]
    cell2 = net2.cells[i]
    copy_preprocess(cell1.preprocess0, cell2.preprocess0)
    copy_preprocess(cell1.preprocess1, cell2.preprocess1)
    if i in [num_layers // 3, 2 * num_layers // 3]:
        copy_sep_conv(cell1._ops[0][1], cell2._ops[1])
        copy_preprocess(cell1._ops[1][0], cell2._ops[2])
        copy_dil_conv(cell1._ops[2][1], cell2._ops[5])
        copy_dil_conv(cell1._ops[3][0], cell2._ops[6])
        copy_dil_conv(cell1._ops[3][1], cell2._ops[7])
    else:
        copy_sep_conv(cell1._ops[0][0], cell2._ops[0])
        copy_sep_conv(cell1._ops[0][1], cell2._ops[1])
        copy_sep_conv(cell1._ops[1][0], cell2._ops[2])
        copy_sep_conv(cell1._ops[1][1], cell2._ops[3])
        copy_sep_conv(cell1._ops[2][0], cell2._ops[4])
        copy_sep_conv(cell1._ops[2][1], cell2._ops[5])
        copy_sep_conv(cell1._ops[3][1], cell2._ops[7])

copy_linear(net1.classifier.layers[1], net2.classifier)

if weight_decay:
    print(tf.add_n(net1.losses))

for _ in range(1):
    x1 = tf.random.normal([2, 32, 32, 3])
    x2 = torch.from_numpy(np.transpose(x1.numpy(), [0, 3, 1, 2]))
    y1 = tf.random.uniform((2,), 0, 10, dtype=tf.int32)
    y2 = torch.from_numpy(y1.numpy()).to(torch.int64)
    # y1 = tf.one_hot(y1, 10)
    # p1 = net1(x1, training=False)[0]
    #
    # net2.eval()
    # with torch.no_grad():
    #     p2 = net2(x2)[0]
    #
    # d1 = p1.numpy() - p2.numpy()

    with tf.GradientTape() as tape:
        p1 = net1(x1, training=True)
        loss1 = criterion1(y1, p1)
        loss1 = tf.reduce_mean(loss1)
        if weight_decay != 0:
            loss1 = loss1 + tf.add_n(net1.losses)
    grads = tape.gradient(loss1, net1.trainable_variables)
    # grads = tf.clip_by_global_norm(grads, 5.0)[0]
    optimizer1.apply_gradients(zip(grads, net1.trainable_variables))

    net2.train()
    net2.zero_grad()
    p2 = net2(x2)[0]
    loss2 = criterion2(p2, y2)
    loss2.backward()
    # torch.nn.utils.clip_grad_norm_(net2.parameters(), 5.0)
    optimizer2.step()

    d2 = p1.numpy() - p2.detach().numpy()
    if weight_decay:
        print(tf.add_n(net1.losses))
    print(d2.mean(), d2.std())