from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model


def op2model(op):
    def model_func(input_shape, *args, **kwargs):
        inputs = Input(input_shape)
        model = Model(inputs=inputs, outputs=op(inputs, *args, **kwargs))
        return model

    return model_func


def replace_layer(layers, input_shape, layer_names, create_layer_funcs):
    if not isinstance(create_layer_funcs, (tuple, list)):
        create_layer_funcs = [create_layer_funcs] * len(layer_names)
    else:
        assert len(layer_names) == len(create_layer_funcs)

    input_layers_of = defaultdict(list)
    new_output_tensor_of = defaultdict(list)

    inputs = Input(input_shape)
    for layer in layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            input_layers_of[layer_name].append(layer.name)
    new_output_tensor_of[layers[0].name].append(inputs)
    #     return input_layers_of, new_output_tensor_of
    for layer in layers[1:]:
        layer_input = [l
                       for layer_aux in input_layers_of[layer.name]
                       for l in new_output_tensor_of[layer_aux]
                       ]

        if len(layer_input) == 1:
            layer_input = layer_input[0]
        print(layer_input)
        if layer.name in layer_names:
            x = layer_input
            new_layer = create_layer_funcs[layer_names.index(layer.name)](layer)
            x = new_layer(x)
        else:
            x = layer(layer_input)

        new_output_tensor_of[layer.name].append(x)

    return Model(inputs=inputs, outputs=x)


def decimate(tensor, m):
    for d in range(tensor.ndim):
        if m[d] is not None:
            indices = tf.range(0, tensor.shape[d], m[d])
            tensor = tf.gather(tensor, indices, axis=d)
    return tensor