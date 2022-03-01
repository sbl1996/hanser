from PIL import Image
import numpy as np
import tensorflow as tf

tflite_model_path = "/Users/hrvvi/Downloads/lite-model_imagenet_mobilenet_v3_large_100_224_classification_5_metadata_1.tflite"

img = Image.open("/Users/hrvvi/Downloads/images/cat1.jpeg")
img = img.resize((457, 256), Image.BILINEAR).crop((116, 16, 116 + 224, 16 + 224))
x = np.array(img) / 255
# x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.astype(np.float32)


# TF Lite

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
