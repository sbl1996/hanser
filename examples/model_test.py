import tensorflow as tf

from hanser.kerascv.resnet import resnet10
from hanser.models.legacy.layers import get_custom_objects

import tensorflow as tf
import numpy as np




model = resnet10(pretrained=True)
model.save_weights('1.tf')
layers = [
    'features/stage4/unit1/identity_conv/conv',
    'features/stage4/unit1/body/conv1/conv',
]
for l in layers:
    model.get_layer(l).strides = (1, 1)
c = model.get_config()
c['layers'] = c['layers'][:-3]
c['layers'][0]['config']['batch_input_shape'] = (None, 513, 513, 3)
c['output_layers'][0][0] = c['layers'][-1]['name']
# del models

model2 = tf.keras.Model.from_config(c, get_custom_objects())
model2.load_weights('1.tf')