import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer

from hanser.models.darts.operations import FactorizedReduce, ReLUConvBN, OPS
from hanser.models.darts.genotypes import get_primitives, Genotype
from hanser.models.layers import Norm, Conv2d, GlobalAvgPool, Linear, Pool2d


def channel_shuffle(x, groups):
    b, h, w, c = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = c // groups

    x = tf.reshape(x, [b, h, w, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [b, h, w, c])
    return x


class MixedOp(Layer):

    def __init__(self, C, stride, k, name):
        super().__init__(name=name)
        self.stride = stride
        self.mp = Pool2d(2, 2, type='max', name='pool')
        self.k = k
        self._ops = []
        self._channels = C // k
        for i, primitive in enumerate(get_primitives()):
            if 'pool' in primitive:
                op = Sequential([
                    OPS[primitive](self._channels, stride, name='pool'),
                    Norm(self._channels, name='norm')
                ], name=f'op{i+1}')
            else:
                op = OPS[primitive](self._channels, stride, name=f'op{i+1}')
            self._ops.append(op)

    def call(self, inputs):
        x, alphas = inputs
        x1 = x[:, :, :, :self._channels]
        x2 = x[:, :, :, self._channels:]
        x1 = tf.add_n([alphas[i] * op(x1) for i, op in enumerate(self._ops)])
        if self.stride == 1:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = tf.concat([x1, self.mp(x2)], axis=-1)
        x = channel_shuffle(x, self.k)
        return x


class Cell(Layer):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k, name):
        super().__init__(name=name)
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, name='preprocess0')
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, name='preprocess0')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, name='preprocess1')
        self._steps = steps
        self._multiplier = multiplier
        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, k, name=f'mixop_{j}_{i+2}')
                self._ops.append(op)

    def call(self, inputs):
        s0, s1, alphas, betas = inputs
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = tf.add_n([betas[offset + j] * self._ops[offset + j]([h, alphas[offset + j]]) for j, h in enumerate(states)])
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self._multiplier:], axis=-1)


def beta_softmax(betas, steps):
    beta_list = []
    offset = 0
    for i in range(steps):
        beta_list.append(
            tf.nn.softmax(betas[offset:(offset + i + 2)], axis=0))
        offset += i + 2
    betas = tf.concat(beta_list, axis=0)
    return betas


class Network(Model):

    def __init__(self, C=16, layers=8, steps=4, multiplier=4, stem_multiplier=3, k=4, num_classes=10):
        super().__init__(name='network')
        self._C = C
        self._steps = steps
        self._multiplier = multiplier
        self._k = k

        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='default', name='stem')

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, k, name=f'cell{i}')
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pool = GlobalAvgPool(name='global_pool')
        self.fc = Linear(C_prev, num_classes, name='fc')

        k = sum(2 + i for i in range(self._steps))
        num_ops = len(get_primitives())
        self.alphas_normal = self.add_weight(
            'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )
        self.alphas_reduce = self.add_weight(
            'alphas_reduce', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )

        self.betas_normal = self.add_weight(
            'betas_normal', (k,), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )
        self.betas_reduce = self.add_weight(
            'betas_reduce', (k,), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )

    def call(self, x):
        s0 = s1 = self.stem(x)

        alphas_reduce = tf.nn.softmax(self.alphas_reduce, axis=-1)
        alphas_normal = tf.nn.softmax(self.alphas_normal, axis=-1)

        betas_reduce = beta_softmax(self.betas_reduce, self._steps)
        betas_normal = beta_softmax(self.betas_normal, self._steps)

        for cell in self.cells:
            alphas = alphas_reduce if cell.reduction else alphas_normal
            betas = betas_reduce if cell.reduction else betas_normal
            s0, s1 = s1, cell([s0, s1, alphas, betas])
        x = self.global_pool(s1)
        logits = self.fc(x)
        return logits

    def model_parameters(self):
        return self.trainable_variables[:-4]

    def arch_parameters(self):
        return self.trainable_variables[-4:]

    def genotype(self):
        PRIMITIVES = get_primitives()

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return w[i], PRIMITIVES[i]

        def _parse(alphas, betas):
            gene = []
            start = 0
            for i in range(self._steps):
                end = start + i + 2
                W = alphas[start:end] * betas[start:end]
                edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
                for j in edges:
                    gene.append((get_op(W[j])[1], j))
                start = end
            return gene

        gene_normal = _parse(
            tf.nn.softmax(self.alphas_normal, axis=-1).numpy(),
            beta_softmax(self.betas_normal, self._steps).numpy(),
        )
        gene_reduce = _parse(
            tf.nn.softmax(self.alphas_reduce, axis=-1).numpy(),
            beta_softmax(self.betas_reduce, self._steps).numpy(),
        )

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

