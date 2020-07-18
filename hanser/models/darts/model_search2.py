import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
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
        return tf.add_n([weights[i] * op(x) for i, op in enumerate(self._ops)])


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
        self._num_ops = len(self._ops)

    def call(self, inputs):
        s0, s1, weights = inputs
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        s2 = tf.add_n([
            self._ops[0]([s0, weights[0]]),
            self._ops[1]([s1, weights[1]]),
        ])

        s3 = tf.add_n([
            self._ops[2]([s0, weights[2]]),
            self._ops[3]([s1, weights[3]]),
            self._ops[4]([s2, weights[4]]),
        ])

        s4 = tf.add_n([
            self._ops[5]([s0, weights[5]]),
            self._ops[6]([s1, weights[6]]),
            self._ops[7]([s2, weights[7]]),
            self._ops[8]([s3, weights[8]]),
        ])

        s5 = tf.add_n([
            self._ops[9]([s0, weights[9]]),
            self._ops[10]([s1, weights[10]]),
            self._ops[11]([s2, weights[11]]),
            self._ops[12]([s3, weights[12]]),
            self._ops[13]([s4, weights[13]]),
        ])
        return tf.concat([s2, s3, s4, s5], axis=-1)


class Network(Model):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10):
        super().__init__(name='network')
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

        k = sum(2 + i for i in range(4))
        num_ops = len(get_primitives())
        self.alphas_normal = self.add_weight(
            'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )
        self.alphas_reduce = self.add_weight(
            'alphas_reduce', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )

    def call(self, x):
        s0 = s1 = self.stem(x)
        weights_reduce = tf.nn.softmax(self.alphas_reduce, axis=-1)
        weights_normal = tf.nn.softmax(self.alphas_normal, axis=-1)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell([s0, s1, weights])
        x = self.global_pool(s1)
        logits = self.fc(x)
        return logits
    #
    # def build(self, input_shape):
    #     k = sum(2 + i for i in range(4))
    #     num_ops = len(get_primitives())
    #     self.alphas_normal = self.add_weight(
    #         'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
    #     )
    #     self.alphas_reduce = self.add_weight(
    #         'alphas_reduce', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
    #     )
    #     super().build(input_shape)

    def arch_parameters(self):
        return self.trainable_variables[-2:]

    def model_parameters(self):
        return self.trainable_variables[:-2]

    def genotype(self):
        PRIMITIVES = get_primitives()

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return w[i], PRIMITIVES[i]

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
                for j in edges:
                    gene.append((get_op(W[j])[1], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())
        gene_reduce = _parse(tf.nn.softmax(self.alphas_reduce, axis=-1).numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype