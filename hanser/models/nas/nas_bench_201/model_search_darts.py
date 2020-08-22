import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer

from hanser.models.nas.operations import ReLUConvBN, OPS
from hanser.models.nas.genotypes import get_primitives
from hanser.models.layers import Norm, Conv2d, GlobalAvgPool, Linear, Pool2d, Act
from hanser.models.modules import DropPath


class MixedOp(Layer):

    def __init__(self, C, stride, drop_path, name):
        super().__init__(name=name)
        self.stride = stride
        self._ops = []
        for i, primitive in enumerate(get_primitives()):
            if drop_path:
                op = Sequential([
                    OPS[primitive](C, stride, name='pool'),
                ], name=f'op{i + 1}')
                if 'pool' in primitive:
                    op.add(Norm(C, name='norm'))
                op.add(DropPath(drop_path, name='drop'))
            else:
                if 'pool' in primitive:
                    op = Sequential([
                        OPS[primitive](C, stride, name='pool'),
                        Norm(C, name='norm')
                    ], name=f'op{i + 1}')
                else:
                    op = OPS[primitive](C, stride, name=f'op{i + 1}')
            self._ops.append(op)

    def call(self, inputs):
        x, weights = inputs
        return tf.add_n([weights[i] * op(x) for i, op in enumerate(self._ops)])


class Cell(Layer):

    def __init__(self, steps, C_prev, C, drop_path, name):
        super().__init__(name=name)

        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, name='preprocess1')
        self._steps = steps
        self._ops = []
        for i in range(self._steps):
            for j in range(1 + i):
                op = MixedOp(C, 1, drop_path, name=f'mixop_{j}_{i + 1}')
                self._ops.append(op)

    def call(self, inputs):
        s, weights = inputs
        s = self.preprocess(s)

        states = [s]
        offset = 0
        for i in range(self._steps):
            s = tf.add_n([self._ops[offset + j]([h, weights[offset + j]]) for j, h in enumerate(states)])
            offset += len(states)
            states.append(s)

        return states[-1]


class BasicBlock(Layer):

    def __init__(self, C_prev, C, stride=2, name=None):
        super().__init__(name=name)
        assert stride == 2
        self.conv1 = ReLUConvBN(C_prev, C, 3, stride=stride, name='conv1')
        self.conv2 = ReLUConvBN(C, C, 3, stride=stride, name='conv2')
        self.downsample = Sequential([
            Pool2d(2, 2, type='avg', name='pool'),
            Conv2d(C_prev, C, 1, norm='def'),
        ], name='downsample')
        self.act = Act(name='act')
        self.stride = stride

    def call(self, inputs):
        x, _ = inputs
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.downsample(identity)
        x = self.act(x)
        return x


class Network(Model):

    def __init__(self, C, layers, steps=3, stem_multiplier=3, drop_path=0.6, num_classes=10):
        super().__init__(name='network')
        self._C = C
        self._steps = steps
        self._drop_path = drop_path

        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='default', name='stem')

        C_prev, C_curr = C_curr, C
        self.cells = []
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                cell = BasicBlock(C_prev, C_curr, stride=2, name=f'cell{i}')
            else:
                cell = Cell(steps, C_prev, C_curr, drop_path, name=f'cell{i}')
            self.cells.append(cell)
            C_prev = C_curr

        self.global_pool = GlobalAvgPool(name='global_pool')
        self.fc = Linear(C_prev, num_classes, name='fc')

        k = sum(1 + i for i in range(self._steps))
        num_ops = len(get_primitives())
        self.alphas_normal = self.add_weight(
            'alphas_normal', (k, num_ops), initializer=RandomNormal(stddev=1e-2), trainable=True,
        )

    def call(self, x):
        s = self.stem(x)
        weights_normal = tf.nn.softmax(self.alphas_normal, axis=-1)
        for cell in self.cells:
            s = cell([s, weights_normal])
        x = self.global_pool(s)
        logits = self.fc(x)
        return logits

    def arch_parameters(self):
        return self.trainable_variables[-1:]

    def model_parameters(self):
        return self.trainable_variables[:-1]

    def genotype(self):
        PRIMITIVES = get_primitives()

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return PRIMITIVES[i]

        def _parse(weights):
            genes = []
            start = 0
            for i in range(self._steps):
                gene = []
                end = start + i + 1
                W = weights[start:end]
                for j in range(i + 1):
                    gene.append((get_op(W[j]), j))
                start = end
                genes.append(gene)
            return genes

        gene = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())
        s = "+".join([f"|{'|'.join(f'{op}~{i}' for op, i in ops)}|" for ops in gene])
        return s
