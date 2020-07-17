import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer

from hanser.models.darts.operations import FactorizedReduce, ReLUConvBN, OPS
from hanser.models.darts.genotypes import get_primitives, Genotype
from hanser.models.layers import Norm, Conv2d, GlobalAvgPool, Linear


class MixedOp(Layer):

    def __init__(self, C, stride, name):
        super().__init__(name=name)
        self.stride = stride
        self._ops = []
        for i, primitive in enumerate(get_primitives()):
            if 'pool' in primitive:
                op = Sequential([
                    OPS[primitive](C, stride, name='pool'),
                    Norm(C, name='norm')
                ], name=f'op{i+1}')
            else:
                op = OPS[primitive](C, stride, name=f'op{i+1}')
            self._ops.append(op)

    def call(self, inputs):
        x, weights = inputs
        return sum(weights[i] * op(x) for i, op in enumerate(self._ops))


class Cell(Layer):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, name):
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
                op = MixedOp(C, stride, name=f'mixop_{j}_{i+2}')
                self._ops.append(op)

    def call(self, inputs):
        s0, s1, weights = inputs
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(4):
            s = sum(self._ops[offset + j]([h, weights[offset + j]]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self._multiplier:], axis=-1)


class Network(Model):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10):
        super().__init__()
        self._C = C
        self._steps = steps
        self._multiplier = multiplier

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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, name=f'cell{i}')
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pool = GlobalAvgPool(name='global_pool')
        self.fc = Linear(C_prev, num_classes, name='fc')

    def call(self, x):
        s0 = s1 = self.stem(x)
        weights_reduce = tf.nn.softmax(self.alphas_reduce, axis=-1)
        weights_normal = tf.nn.softmax(self.alphas_normal, axis=-1)
        for i, cell in enumerate(self.cells):
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell([s0, s1, weights])
        x = self.global_pool(s1)
        logits = self.fc(x)
        return logits

    def build(self):
        k = sum(2 + i for i in range(4))
        num_ops = len(get_primitives())
        self.alphas_normal = self.add_weight(
            'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-3)
        )
        self.alphas_reduce = self.add_weight(
            'alphas_reduce', (k, num_ops), initializer=RandomNormal(stddev=1e-3)
        )

    def arch_parameters(self):
        return self.trainable_variables[-2:]

    def model_parameters(self):
        return self.trainable_variables[:-2]
    #
    # def genotype(self):
    #
    #     def _parse(weights):
    #         gene = []
    #         n = 2
    #         start = 0
    #         for i in range(self._steps):
    #             end = start + n
    #             W = weights[start:end].copy()
    #             edges = sorted(range(i + 2),
    #                            key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
    #                     :2]
    #             for j in edges:
    #                 k_best = None
    #                 for k in range(len(W[j])):
    #                     if k != PRIMITIVES.index('none'):
    #                         if k_best is None or W[j][k] > W[j][k_best]:
    #                             k_best = k
    #                 gene.append((PRIMITIVES[k_best], j))
    #             start = end
    #             n += 1
    #         return gene
    #
    #     gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    #     gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
    #
    #     concat = range(2 + self._steps - self._multiplier, self._steps + 2)
    #     genotype = Genotype(
    #         normal=gene_normal, normal_concat=concat,
    #         reduce=gene_reduce, reduce_concat=concat
    #     )
    #     return genotype
    #
