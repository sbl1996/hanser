from PIL import Image
import numpy as np
import tensorflow as tf

save_path = "/Users/hrvvi/Downloads/res2net50"
tflite_model_path = "/Users/hrvvi/Downloads/res2net50.tflite"

img = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
img = img.resize((457, 256), Image.BILINEAR).crop((116, 16, 116 + 224, 16 + 224))
x = np.array(img) / 255
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.astype(np.float32)

# Tensorflow Model

from hanser.models.imagenet.res2net.resnet_vd import resnet50
from hanser.models.utils import load_pretrained_model

net = resnet50()
net.build((None, 224, 224, 3))
load_pretrained_model('res2netvd50', net, with_fc=True)

xt = tf.convert_to_tensor(x)
y1 = net(xt[None]).numpy()[0]
p1 = np.argsort(y1)[-5:][::-1]
v1 = y1[p1]
# [281, 285, 282, 292, 739], [10.137764,  8.578737,  8.302447,  5.743762,  4.512046]

# SavedModel

tf.saved_model.save(net, save_path)

net = tf.saved_model.load(save_path)
xt = tf.convert_to_tensor(x)
y2 = net(xt[None]).numpy()[0]
p2 = np.argsort(y2)[-5:][::-1]
v2 = y2[p2]
# [281, 285, 282, 292, 739], [10.137764,  8.578737,  8.302447,  5.743762,  4.512046]


# TF Lite
converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
tflite_model = converter.convert()
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

interpreter.set_tensor(input_index, x[None])
interpreter.invoke()
y3 = interpreter.get_tensor(output_index)[0]
p3 = np.argsort(y3)[-5:][::-1]
v3 = y3[p3]
# [281, 285, 282, 292, 739], [10.016383 ,  8.439419 ,  8.1703415,  5.7234025,  4.596955 ]

# TFLite
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']

interpreter.set_tensor(input_index, x[None])
interpreter.invoke()
y3 = interpreter.get_tensor(output_index)[0]
p3 = np.argsort(y3)[-5:][::-1]
v3 = y3[p3]