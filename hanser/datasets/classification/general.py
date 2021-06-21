import os

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.datasets.tfds.helper import HGeneratorBasedBuilder, DatasetInfo
from hhutil.io import read_lines


def decode(example):
    image = tf.cast(example['image'], tf.uint8)
    label = tf.cast(example['label'], dtype=tf.int32)
    return image, label


class ImageListBuilder(HGeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    MANUAL_DOWNLOAD_INSTRUCTIONS = """Put it anywhere you like."""

    SPLITS = ['train', 'val', 'test']
    NUM_CLASSES = None
    NAMES = None

    def _info(self) -> DatasetInfo:
        if self.NAMES is None:
            assert isinstance(self.NUM_CLASSES, int)
        else:
            if self.NUM_CLASSES is not None:
                assert len(self.NAMES) == self.NUM_CLASSES
        features = {
            'image': tfds.features.Image(),
            "label": tfds.features.ClassLabel(num_classes=self.NUM_CLASSES, names=self.NAMES),
            "file_name": tfds.features.Text(),
        }

        return DatasetInfo(
            builder=self,
            supervised_keys=("image", "label"),
            features=tfds.features.FeaturesDict(features),
        )

    def _split_generators(self, dl_manager):
        splits = []
        image_dir = os.path.join(dl_manager.manual_dir, "images")
        for k in self.SPLITS:
            label_file = os.path.join(dl_manager.manual_dir, f"{k}_list.txt")
            splits.append(
                tfds.core.SplitGenerator(
                    name=k,
                    gen_kwargs=dict(
                        image_dir=image_dir,
                        label_file=label_file,
                    ))
            )
        return splits

    def _generate_examples(self, image_dir, label_file):
        """Generate examples as dicts.

        Args:
          image_dir: `str`, directory containing the images
          label_file: `str`, directory containing image path and label

        Yields:
          example key and data
        """
        lines = read_lines(label_file)
        for line in lines:
            file_name, label = line.split()
            record = {
                "image": os.path.join(image_dir, file_name),
                "label": int(label),
                "file_name": file_name,
            }
            yield file_name, record