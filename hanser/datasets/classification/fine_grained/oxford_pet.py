import tensorflow_datasets as tfds
from hanser.datasets.classification.general import ImageListBuilder

_DESCRIPTION = """\
The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200
images for each class. The images have large variations in scale, pose and
lighting. All images have an associated ground truth annotation of breed.
"""

_LABEL_CLASSES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
    "British_Shorthair", "chihuahua", "Egyptian_Mau", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "Maine_Coon",
    "miniature_pinscher", "newfoundland", "Persian", "pomeranian", "pug",
    "Ragdoll", "Russian_Blue", "saint_bernard", "samoyed", "scottish_terrier",
    "shiba_inu", "Siamese", "Sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier"
]


class OxfordPet(ImageListBuilder):

    VERSION = tfds.core.Version("1.0.0")
    SPLITS = {
        'train': 8144,
        'test': 8041,
    }
    NAMES = _LABEL_CLASSES
