import tensorflow as tf
from hanser.datasets.imagenet import IMAGENET_CLASSES, make_imagenet_dataset_split
from hanser.transform import random_resized_crop, resize, center_crop, normalize, to_tensor, mixup, mixup_batch, mixup_in_batch


def transform(image, label, training):
    if training:
        image = random_resized_crop(image, 160, scale=(0.05, 1.0), ratio=(0.75, 1.33))
        image = tf.image.random_flip_left_right(image)
    else:
        image = resize(image, 256)
        image = center_crop(image, 224)

    image, label = to_tensor(image, label, label_offset=1)
    image = normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    label = tf.one_hot(label, 1000)
    return image, label

def zip_transform(data1, data2):
    return mixup(data1, data2, alpha=0.2)

def batch_transform(image, label):
    return mixup_in_batch(image, label, alpha=0.2)


train_files = [
    "/Users/hrvvi/Downloads/ILSVRC2012/tfrecords/combined/validation-%05d-of-00128" % i
    for i in range(32)
]

batch_size = 128 * 2
ds_train = make_imagenet_dataset_split(
    batch_size, transform, train_files, split='train',
    cache_dataset=True, cache_decoded_image=False,
    batch_transform=batch_transform)[0]

steps_per_epoch = 10

train_it = iter(ds_train)
for i in range(50000 // 4 // batch_size * 2 + 1):
    x, y = next(train_it)
    print(i, x[0].numpy().mean())

# 2.07GB
# batch 2.10GB
# zip


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# i = 5
# xt = x[i].numpy()
# xt = (xt * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
# plt.imshow(xt.astype(np.uint8))
# print(IMAGENET_CLASSES[np.argmax(y[i])])

