import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer

from hanser.models.modules import DropPath
from hanser.models.layers import Conv2d, Pool2d, GlobalAvgPool, Act, Linear
from hanser.models.darts.operations import FactorizedReduce, ReLUConvBN, OPS


class Cell(Layer):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_path, name):
        super().__init__(name=name)
        self.drop_path = drop_path
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, name='preprocess0')
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, name='preprocess0')
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, name='preprocess1')

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

        self._ops = []
        for i, (name, index) in enumerate(zip(op_names, indices)):
            stride = 2 if reduction and index < 2 else 1
            if self.drop_path and name != 'skip_connect':
                op = Sequential([
                    OPS[name](C, stride, name="op"),
                    DropPath(self.drop_path, name="drop"),
                ], name=f"op{i + 1}")
            else:
                op = OPS[name](C, stride, name=f"op{i}")
            self._ops.append(op)
        self._indices = indices

    def call(self, inputs):
        s0, s1 = inputs
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
            s = h1 + h2
            states.append(s)
        return tf.concat([states[i] for i in self._concat], axis=-1)


class AuxiliaryHeadCIFAR(Layer):

    def __init__(self, C, num_classes, name):
        """assuming input size 8x8"""
        super().__init__(name=name)
        self.features = Sequential([
            Act(name='act0'),
            Pool2d(5, stride=3, padding=0, name='pool'),
            Conv2d(C, 128, 1, norm='def', act='def', name='conv1'),
            Conv2d(128, 768, 2, norm='def', act='def', padding=0, name='conv2'),
        ], name='features')
        self.classifier = Sequential([
            GlobalAvgPool(name='global_pooling'),
            Linear(768, num_classes, name='fc'),
        ], name='classifier')

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DARTS(Model):

    def __init__(self, C, layers, auxiliary, drop_path, num_classes, genotype):
        super().__init__()
        self._num_layers = layers
        self._auxiliary = auxiliary
        self._drop_path = drop_path

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='def', name='stem')

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, drop_path, name=f"cell{i}")
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes, name='aux_head')
        self.classifier = Sequential([
            GlobalAvgPool(name='global_pooling'),
            Linear(C_prev, num_classes, name='fc'),
        ], name='classifier')

    def call(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell((s0, s1))
            if self._auxiliary and i == 2 * self._num_layers // 3:
                logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1)
        if self._auxiliary:
            return logits, logits_aux
        else:
            return logits
