import collections
import json
import os

from absl import logging
import tensorflow as tf

import tensorflow_datasets as tfds

Split = collections.namedtuple(
    'Split', ['name', 'images', 'annotations'])


class CocoConfig(tfds.core.BuilderConfig):
    """BuilderConfig for CocoConfig."""

    def __init__(
        self,
        splits=None,
        has_panoptic=False,
        **kwargs):
        super(CocoConfig, self).__init__(
            version=tfds.core.Version('1.1.0'),
            **kwargs)
        self.splits = splits
        self.has_panoptic = has_panoptic


class CocoVal(tfds.core.GeneratorBasedBuilder):
    """Base MS Coco dataset."""

    BUILDER_CONFIGS = [
        CocoConfig(
            name='2017',
            description="",
            splits=[
                Split(
                    name=tfds.Split.VALIDATION,
                    images='val2017',
                    annotations='annotations_trainval2017',
                ),
            ],
        ),
    ]

    def _info(self):

        features = {
            'image': tfds.features.Image(encoding_format='jpeg'),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int64,
        }

        features.update({
            'objects': tfds.features.Sequence({
                'id': tf.int64,
                # Coco has unique id for each annotation. The id can be used for
                # mapping panoptic image to semantic segmentation label.
                'area': tf.int64,
                'bbox': tfds.features.BBoxFeature(),
                # Coco has 91 categories but only 80 are present in the dataset
                'label': tfds.features.ClassLabel(num_classes=80),
                'is_crowd': tf.bool,
            }),
        })

        return tfds.core.DatasetInfo(
            builder=self,
            description="",
            features=tfds.features.FeaturesDict(features),
            homepage='http://cocodataset.org/#home',
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # Merge urls from all splits together
        urls = {}
        for split in self.builder_config.splits:
            urls['{}_images'.format(split.name)] = 'zips/{}.zip'.format(split.images)
            urls['{}_annotations'.format(split.name)] = 'annotations/{}.zip'.format(
                split.annotations)

        # DownloadManager memoize the url, so duplicate urls will only be downloaded
        # once.
        root_url = 'http://images.cocodataset.org/'
        extracted_paths = dl_manager.download_and_extract({
            key: root_url + url for key, url in urls.items()
        })

        splits = []
        for split in self.builder_config.splits:
            image_dir = extracted_paths['{}_images'.format(split.name)]
            annotations_dir = extracted_paths['{}_annotations'.format(split.name)]

            splits.append(tfds.core.SplitGenerator(
                name=split.name,
                gen_kwargs=dict(
                    image_dir=image_dir,
                    annotation_dir=annotations_dir,
                    split_name=split.images,
                ),
            ))
        return splits

    def _generate_examples(
        self,
        image_dir,
        annotation_dir,
        split_name):
        """Generate examples as dicts.

        Args:
          image_dir: `str`, directory containing the images
          annotation_dir: `str`, directory containing annotations
          split_name: `str`, <split_name><year> (ex: train2014, val2017)
          annotation_type: `AnnotationType`, the annotation format (NONE, BBOXES,
            PANOPTIC)
          panoptic_dir: If annotation_type is PANOPTIC, contains the panoptic
            image directory

        Yields:
          example key and data
        """

        instance_filename = 'instances_{}.json'

        # Load the annotations (label names, images metadata,...)
        instance_path = os.path.join(
            annotation_dir,
            'annotations',
            instance_filename.format(split_name),
        )
        coco_annotation = CocoAnnotationBBoxes(instance_path)

        categories = coco_annotation.categories

        images = coco_annotation.images

        # TODO(b/121375022): ClassLabel names should also contains 'id' and
        # and 'supercategory' (in addition to 'name')
        # Warning: As Coco only use 80 out of the 91 labels, the c['id'] and
        # dataset names ids won't match.
        if self.builder_config.has_panoptic:
            objects_key = 'panoptic_objects'
        else:
            objects_key = 'objects'
        self.info.features[objects_key]['label'].names = [
            c['name'] for c in categories
        ]
        # TODO(b/121375022): Conversion should be done by ClassLabel
        categories_id2name = {c['id']: c['name'] for c in categories}

        # Iterate over all images
        annotation_skipped = 0
        for image_info in sorted(images, key=lambda x: x['id']):
            instances = coco_annotation.get_annotations(img_id=image_info['id'])

            if not instances:
                annotation_skipped += 1

            def build_bbox(x, y, width, height):
                return tfds.features.BBox(
                    ymin=y / image_info['height'],
                    xmin=x / image_info['width'],
                    ymax=(y + height) / image_info['height'],
                    xmax=(x + width) / image_info['width'],
                )
                # pylint: enable=cell-var-from-loop

            example = {
                'image': os.path.join(image_dir, split_name, image_info['file_name']),
                'image/filename': image_info['file_name'],
                'image/id': image_info['id'],
                objects_key: [{  # pylint: disable=g-complex-comprehension
                    'id': instance['id'],
                    'area': instance['area'],
                    'bbox': build_bbox(*instance['bbox']),
                    'label': categories_id2name[instance['category_id']],
                    'is_crowd': bool(instance['iscrowd']),
                } for instance in instances]
            }

            yield image_info['file_name'], example

        logging.info(
            '%d/%d images do not contains any annotations',
            annotation_skipped,
            len(images),
        )


class CocoAnnotation(object):
    """Coco annotation helper class."""

    def __init__(self, annotation_path):
        with tf.io.gfile.GFile(annotation_path) as f:
            data = json.load(f)
        self._data = data

    @property
    def categories(self):
        """Return the category dicts, as sorted in the file."""
        return self._data['categories']

    @property
    def images(self):
        """Return the image dicts, as sorted in the file."""
        return self._data['images']

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        raise NotImplementedError  # AnotationType.NONE don't have annotations


class CocoAnnotationBBoxes(CocoAnnotation):
    """Coco annotation helper class."""

    def __init__(self, annotation_path):
        super(CocoAnnotationBBoxes, self).__init__(annotation_path)

        img_id2annotations = collections.defaultdict(list)
        for a in self._data['annotations']:
            img_id2annotations[a['image_id']].append(a)
        self._img_id2annotations = {
            k: list(sorted(v, key=lambda a: a['id']))
            for k, v in img_id2annotations.items()
        }

    def get_annotations(self, img_id):
        """Return all annotations associated with the image id string."""
        # Some images don't have any annotations. Return empty list instead.
        return self._img_id2annotations.get(img_id, [])
