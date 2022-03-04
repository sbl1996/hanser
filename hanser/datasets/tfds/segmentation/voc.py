from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_VOC_CITATION = """\
@misc{{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {{PASCAL}} {{V}}isual {{O}}bject {{C}}lasses {{C}}hallenge 2012 {{(VOC2012)}} {{R}}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}}
"""

_SBD_CITATION = """\
@InProceedings{{BharathICCV2011,
    author = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik",
    title = "Semantic Contours from Inverse Detectors",
    booktitle = "International Conference on Computer Vision (ICCV)",
    year = "2011"}}
"""

_VOC_DESCRIPTION = """"""


_VOC_LABELS = (
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
)


_DOWNLOAD_URL = {
    "trainval": "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
    "sbd": "https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
}


class VOCSeg(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_VOC_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(encoding_format="jpeg"),
                "annotation": tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8),
            }),
            supervised_keys=("image", "annotation"),
            homepage="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/",
            citation=_VOC_CITATION + _SBD_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_paths = dl_manager.download_and_extract(_DOWNLOAD_URL)
        voc_root = Path(dl_paths["trainval"]) / "VOCdevkit" / "VOC2012"
        sbt_root = Path(dl_paths["sbd"]) / "benchmark_RELEASE" / "dataset"
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "images_dir":
                        voc_root / "JPEGImages",
                    "annotations_dir":
                        voc_root / "SegmentationClass",
                    "id_files":
                        [voc_root / "ImageSets" / "Segmentation" / "train.txt"],
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "images_dir":
                        voc_root / "JPEGImages",
                    "annotations_dir":
                        voc_root / "SegmentationClass",
                    "id_files":
                        [voc_root / "ImageSets" / "Segmentation" / "val.txt"],
                },
            ),
            tfds.core.SplitGenerator(
                name='sbd',
                gen_kwargs={
                    "images_dir":
                        sbt_root / "img",
                    "annotations_dir":
                        sbt_root / "cls",
                    "id_files": [
                        sbt_root / "train.txt", sbt_root / "val.txt"],
                    "exclude_id_files": [
                        voc_root / "ImageSets" / "Segmentation" / "train.txt",
                        voc_root / "ImageSets" / "Segmentation" / "val.txt",
                    ],
                    "mat_annotation": True,
                },
            ),
        ]

    def _generate_examples(self, images_dir, annotations_dir, id_files, exclude_id_files=None, mat_annotation=False):
        exclude_ids = set()
        if exclude_id_files:
            for id_file in exclude_id_files:
                with open(id_file, "r") as f:
                    for line in f:
                        exclude_ids.add(line.strip())
        ids = set()
        for id_file in id_files:
            with open(id_file, "r") as f:
                for line in f:
                    id_ = line.strip()
                    if id_ not in exclude_ids:
                        ids.add(id_)
        for example_id in ids:
            image_path = images_dir / "{}.jpg".format(example_id)
            if mat_annotation:
                fp = annotations_dir / "{}.mat".format(example_id)
                record = tfds.core.lazy_imports.scipy.io.loadmat(fp, struct_as_record=True)
                class_mask = np.expand_dims(record["GTcls"]["Segmentation"][0][0], axis=-1)
            else:
                fp = annotations_dir / "{}.png".format(example_id)
                image = tfds.core.lazy_imports.PIL_Image.open(fp)
                class_mask = np.expand_dims(np.array(image, dtype=np.uint8), axis=-1)
            yield example_id, {
                "image": image_path,
                "annotation": class_mask,
            }