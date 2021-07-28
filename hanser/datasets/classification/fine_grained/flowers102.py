import tensorflow_datasets as tfds

from hanser.datasets.classification.general import ImageListBuilder

class Flowers102(ImageListBuilder):

    VERSION = tfds.core.Version("1.0.0")
    SPLITS = {
        'train': 6149,
        'val': 1020,
        'test': 1020,
    }
    NUM_CLASSES = 102