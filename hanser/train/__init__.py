from tensorflow.python.keras.layers import Conv2D, Dense
from tensorflow.python.keras.regularizers import l2


def l2_regularizer(model, alpha):
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(l2(alpha)(layer.bias))
    return model