import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer

from hanser.models.nas.operations import FactorizedReduce, ReLUConvBN, OPS
from hanser.models.nas.genotypes import get_primitives, Genotype
from hanser.models.layers import Norm, Conv2d, GlobalAvgPool, Linear
from hanser.models.modules import DropPath


class MixedOp(Layer):

    def __init__(self, C, stride, drop_path):
        super().__init__()
        self.stride = stride
        self._ops = []
        for i, primitive in enumerate(get_primitives()):
            if drop_path:
                op = Sequential([
                    OPS[primitive](C, stride),
                ])
                if 'pool' in primitive:
                    op.add(Norm(C))
                op.add(DropPath(drop_path))
            else:
                op = OPS[primitive](C, stride)
                if 'pool' in primitive:
                    op = Sequential([
                        op,
                        Norm(C)
                    ])
            self._ops.append(op)

    def call(self, inputs):
        x, weights = inputs
        return tf.add_n([weights[i] * op(x) for i, op in enumerate(self._ops)])


class Cell(Layer):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_path):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, drop_path)
                self._ops.append(op)

    def call(self, inputs):
        s0, s1, weights = inputs
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = tf.add_n([self._ops[offset + j]([h, weights[offset + j]]) for j, h in enumerate(states)])
            offset += len(states)
            states.append(s)

        return tf.concat(states[-self._multiplier:], axis=-1)


class Network(Model):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, drop_path=0.6, num_classes=10):
        super().__init__()
        self._C = C
        self._steps = steps
        self._multiplier = multiplier
        self._drop_path = drop_path

        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='default')

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, drop_path)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = GlobalAvgPool()
        self.classifier = Linear(C_prev, num_classes)

        k = sum(2 + i for i in range(self._steps))
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
        x = self.avg_pool(s1)
        logits = self.classifier(x)
        return logits

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
            start = 0
            for i in range(self._steps):
                end = start + i + 2
                W = weights[start:end]
                edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
                for j in edges:
                    gene.append((get_op(W[j])[1], j))
                start = end
            return gene

        gene_normal = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())
        gene_reduce = _parse(tf.nn.softmax(self.alphas_reduce, axis=-1).numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

