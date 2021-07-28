import tensorflow_datasets as tfds
from hanser.datasets.classification.general import ImageListBuilder

class StanfordCars(ImageListBuilder):

    VERSION = tfds.core.Version("1.0.0")
    SPLITS = {
        'train': 8144,
        'test': 8041,
    }
    NUM_CLASSES = 196