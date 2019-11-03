from tensorflow.keras.layers import Conv2D, Dense, DepthwiseConv2D
from tensorflow.keras.regularizers import l2


def l2_regularizer(model, alpha):
    for layer in model.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(alpha)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(alpha)(layer.kernel))

        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(l2(alpha)(layer.bias))
    return model