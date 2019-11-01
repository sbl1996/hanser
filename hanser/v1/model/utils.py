from tensorflow.keras.layers import Input
from tensorflow.keras import Model


def op2model(op):
    def model_func(input_shape, *args, **kwargs):
        inputs = Input(input_shape)
        model = Model(inputs=inputs, outputs=op(inputs, *args, **kwargs))
        return model

    return model_func
