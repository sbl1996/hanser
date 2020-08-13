import tensorflow as tf

from hanser.models.nas.operations import OPS

ops = [
    OPS["skip_connect"](16, 1, "op1"),
    OPS["sep_conv_3x3"](16, 1, "op2"),
    OPS["none"](16, 1, "op3"),
]

@tf.function
def mix_op(x, weights):
    return sum(weights[i] * op(x) for i, op in enumerate(ops))

print(tf.autograph.to_code(mix_op.python_function))