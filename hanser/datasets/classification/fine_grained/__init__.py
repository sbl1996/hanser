from typing import Type
from hanser.datasets.classification.general import ImageListBuilder
from hanser.datasets.classification.fine_grained.fgvc_aircraft import FGVCAircraft
from hanser.datasets.classification.fine_grained.flowers102 import Flowers102
from hanser.datasets.classification.fine_grained.oxford_pet import OxfordPet
from hanser.datasets.classification.fine_grained.stanford_cars import StanfordCars

def get_num_classes(dataset_cls: Type[ImageListBuilder]):
    return dataset_cls.NUM_CLASSES or len(dataset_cls.NAMES)