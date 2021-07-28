import tensorflow_datasets as tfds

from hanser.datasets.classification.general import ImageListBuilder

class FGVCAircraft(ImageListBuilder):

    VERSION = tfds.core.Version("1.0.0")
    SPLITS = {
        'train': 3334,
        'val': 3333,
        'test': 3333,
    }
    NUM_CLASSES = 100
