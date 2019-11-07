import tensorflow as tf

from hanser.kerascv.resnet import resnet10
from hanser.model.layers import ReflectionPad2D
from hanser.model.layers import get_custom_objects

model = resnet10(pretrained=True)
c = model.get_config()
m = tf.keras.Model.from_config(c, get_custom_objects())
