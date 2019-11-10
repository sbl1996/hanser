import random
import colorsys

import tensorflow as tf
from hanser.ops import to_float, to_int


def get_random_scale(height, width, output_size, scale_min, scale_max):
    random_scale_factor = tf.random.uniform((), scale_min, scale_max)
    scaled_size = to_int(random_scale_factor * output_size)

    max_size = to_float(tf.maximum(height, width))
    img_scale = to_float(scaled_size) / max_size

    scaled_height = to_int(to_float(height) * img_scale)
    scaled_width = to_int(to_float(width) * img_scale)

    offset_y = to_float(scaled_height - output_size)
    offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform((), 0, 1)
    offset_y = to_int(offset_y)

    offset_x = to_float(scaled_width - output_size)
    offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform((), 0, 1)
    offset_x = to_int(offset_x)

    return img_scale, scaled_height, scaled_width, offset_x, offset_y


def scale_box(boxes, scales):
    return tf.reshape(tf.reshape(boxes, [-1, 2, 2]) * scales, [-1, 4])


def resize_with_pad(image, boxes, target_height, target_width):
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    img_scale = tf.minimum(
        to_float(target_height) / to_float(height),
        to_float(target_width) / to_float(width),
    )
    scaled_height = to_int(to_float(height) * img_scale)
    scaled_width = to_int(to_float(width) * img_scale)
    boxes = scale_box(boxes, to_float([scaled_height, scaled_width]) / to_float([target_height, target_width]))
    image = tf.image.resize(image, (scaled_height,  scaled_width))
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height, target_width)
    return image, boxes


def resize_and_crop_image(image, scaled_height, scaled_width, output_size, offset_x, offset_y):
    image = tf.compat.v1.image.resize_bilinear(image[None], (scaled_height, scaled_width), align_corners=True)[0]
    image = image[
        offset_y:offset_y + output_size,
        offset_x:offset_x + output_size,
    ]
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, output_size, output_size
    )
    return image


def resize_and_crop_boxes(boxes, classes, scaled_height, scaled_width, output_size, offset_x, offset_y):
    boxes = tf.reshape(boxes, [-1, 2, 2])
    boxes = boxes * to_float([scaled_height, scaled_width]) - to_float([offset_y, offset_x])
    boxes = tf.reshape(boxes, [-1, 4])
    boxes = tf.clip_by_value(boxes, 0, output_size)
    indices = (boxes[:, 0] != boxes[:, 2]) & (boxes[:, 1] != boxes[:, 3])
    boxes = boxes[indices]
    classes = classes[indices]
    return boxes, classes


# def random_colors(n):
#     step = 256 / n
#     colors = tf.random.uniform([n, 4]) * 256
#     colors = tf.range(n, dtype=tf.float32)[:, None] * step + colors
#     return colors


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def draw_bboxes(img, boxes, classes=None, categories=None, fontsize=8, linewidth=2, colors=None, label_offset=16, figsize=(10, 10)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not colors:
        if categories:
            colors = random_colors(len(categories))
        else:
            colors = ['w' for _ in range(100)]

    boxes = boxes.reshape(-1, 2, 2)[..., ::-1].reshape(-1, 4)
    # boxes = (boxes.reshape(-1, 2, 2) * np.array(img.shape[:2]))[..., ::-1].reshape(-1, 4)
    boxes[:, 2:] -= boxes[:, :2]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    for box, cls in zip(boxes, classes):
        color = colors[cls]
        rect = Rectangle(box[:2], box[2], box[3], linewidth=linewidth,
                         alpha=0.7, edgecolor=color, facecolor='none')

        ax.add_patch(rect)
        if categories:
            text = "%s" % categories[cls]
            # text = "%s %.2f" % (categories[cls], ann['score'])
            ax.text(box[0], box[1] + label_offset, text,
                    color=color, size=fontsize, backgroundcolor="none")
    return fig, ax


#
# def draw_bboxes(images, bboxes):
#     rank = images.ndim
#     assert bboxes.ndim == rank - 1
#     if rank == 3:
#         images = images[None]
#         bboxes = bboxes[None]
#     images = tf.cast(images, tf.float32)
#     images = tf.image.draw_bounding_boxes(images, bboxes, random_colors(16))
#     images = tf.cast(images, tf.uint8).numpy()
#     if rank == 3:
#         images = images[0]
#     return images

VOC_CATEGORIES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]