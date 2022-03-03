import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@inproceedings{zhou2017scene,
title={Scene Parsing through ADE20K Dataset},
author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
year={2017}
}
"""

_DESCRIPTION = """"""

_DOWNLOAD_URL = {
    "trainval": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    "test": "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
}


class ADE_20K(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(encoding_format="jpeg"),
                "annotation": tfds.features.Image(encoding_format="png")
            }),
            supervised_keys=("image", "annotation"),
            homepage="http://sceneparsing.csail.mit.edu/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_path = dl_manager.download_and_extract(_DOWNLOAD_URL['trainval'])
        root_dir = os.path.join(dl_path, "ADEChallengeData2016")
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir_path":
                        os.path.join(root_dir, "images/training"),
                    "annotations_dir_path":
                        os.path.join(root_dir, "annotations/training")
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "images_dir_path":
                        os.path.join(root_dir, "images/validation"),
                    "annotations_dir_path":
                        os.path.join(root_dir, "annotations/validation")
                },
            ),
        ]

    def _generate_examples(self, images_dir_path, annotations_dir_path):
        for image_file in tf.io.gfile.listdir(images_dir_path):
            image_id = os.path.split(image_file)[1].split(".")[0]
            yield image_id, {
                "image":
                    os.path.join(images_dir_path, "{}.jpg".format(image_id)),
                "annotation":
                    os.path.join(annotations_dir_path, "{}.png".format(image_id))
            }
