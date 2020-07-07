# dcnn = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
#
# network_function = tf.function(lambda inputs: dcnn(inputs),
#                                [tf.TensorSpec(tf.TensorShape([None, 224, 224, 3]))])
#
# outputs = list(map(lambda tname: network_function.get_concrete_function().graph.get_tensor_by_name(tname), [
#     'vgg16/block3_pool/MaxPool:0',
#     'vgg16/block4_pool/MaxPool:0',
#     'vgg16/block5_pool/MaxPool:0'
# ]))