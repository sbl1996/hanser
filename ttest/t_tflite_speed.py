import numpy as np
import tensorflow as tf
# import tflite_runtime.interpreter as tflite

class Interpreter:

    def __init__(self, tflite_model_path):
        self.model_path = tflite_model_path
        self.interpreter = tf.lite.Interpreter(tflite_model_path)
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.input_index = input_details[0]['index']
        self.output_index = output_details[0]['index']

    def infer(self, x):
        self.interpreter.set_tensor(self.input_index, x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)

x = np.random.rand(1, 224, 224, 3).astype(np.float32)

interpreter1 = Interpreter("/Users/hrvvi/Downloads/mobilenetv3_large.tflite")
interpreter2 = Interpreter("/Users/hrvvi/Downloads/lite-mobilenet_v3_large_100_224.tflite")


interpreter1 = Interpreter("mobilenetv3_large.tflite")
interpreter2 = Interpreter("lite-mobilenet_v3_large_100_224.tflite")
