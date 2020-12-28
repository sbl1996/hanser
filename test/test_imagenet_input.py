import tensorflow as tf
from hanser.datasets.imagenet import make_imagenet_dataset
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor

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
ds_train, ds_eval = make_imagenet_dataset(data_dir, 256, 512, transform)
