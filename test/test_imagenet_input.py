import tensorflow as tf
from hanser.datasets.imagenet import make_imagenet_dataset, IMAGENET_CLASSES
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor
from hhutil.io import eglob

def transform(image, label, training):
    if training:
        image = random_resized_crop(image, 224, scale=(0.05, 1.0), ratio=(0.75, 1.33))
        image = tf.image.random_flip_left_right(image)
    else:
        image = resize(image, 256)
        image = center_crop(image, 224)

    image, label = to_tensor(image, label, label_offset=1)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    label = tf.one_hot(label, 1000)
    return image, label

data_dir = "/Users/hrvvi/Downloads/ILSVRC2012/tfrecords/combined"
# train_files = [
#     "/Users/hrvvi/Downloads/train-00000-of-01024",
# ]
train_files = [ str(p) for p in eglob(data_dir, "validation-*") ]
eval_files = [
    "/Users/hrvvi/Downloads/ILSVRC2012/tfrecords/combined/validation-00000-of-00128",
    "/Users/hrvvi/Downloads/ILSVRC2012/tfrecords/combined/validation-00042-of-00128"
]
# eval_files = [ str(p) for p in eglob(data_dir, "validation-*") ]
ds_train, ds_eval, steps_per_epoch, eval_steps = make_imagenet_dataset(
    128, 256, transform, train_files=train_files, eval_files=eval_files, cache_decoded_image=False)

train_it = iter(ds_train)
for i in range(1000):
    x, y = next(train_it)
    print(i, x[0].numpy().mean())

import numpy as np
import matplotlib.pyplot as plt

i = 5
xt = x[i].numpy()
xt = (xt * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
plt.imshow(xt.astype(np.uint8))
print(IMAGENET_CLASSES[np.argmax(y[i])])