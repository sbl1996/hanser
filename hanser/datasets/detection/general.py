import os
import json

from toolz import groupby

import tensorflow as tf
import tensorflow_datasets as tfds

from hanser.datasets.tfds.helper import HGeneratorBasedBuilder, DatasetInfo

def decode(example):
    image_id = example['image/id']
    image = tf.cast(example['image'], tf.float32)
    objects = {
        "gt_bbox": example['objects']['bbox'],
        'gt_label': tf.cast(example['objects']['label'] + 1, tf.int32),
    }
    return image, objects, image_id


class CocoBuilder(HGeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """Put it anywhere you like."""

    LABEL_OFFSET = 0
    SPLITS = ['train', 'val', 'test']
    NUM_CLASSES = None

    def _info(self) -> DatasetInfo:
        """Returns the dataset metadata."""
        assert isinstance(self.NUM_CLASSES, int)
        features = {
            # Images can have variable shape
            'image': tfds.features.Image(encoding_format='png'),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int32,
            'objects': tfds.features.Sequence({
                'id': tf.int64,
                'area': tf.int64,
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(num_classes=self.NUM_CLASSES),
                'is_crowd': tf.bool,
            }),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(features),
        )

    def _split_generators(self, dl_manager):
        splits = []
        for k in self.SPLITS:
            image_dir = os.path.join(dl_manager.manual_dir, k)
            ann_file = os.path.join(dl_manager.manual_dir, k + '.json')
            splits.append(
                tfds.core.SplitGenerator(
                    name=k,
                    gen_kwargs=dict(
                        image_dir=image_dir,
                        ann_file=ann_file,
                    ))
            )
        return splits

    def _generate_examples(self, image_dir, ann_file):
        """Generate examples as dicts.

        Args:
          image_dir: `str`, directory containing the images
          ann_file: `str`, directory containing annotations

        Yields:
          example key and data
        """
        with open(ann_file) as f:
            data = json.load(f)
        img_id2anns = groupby('image_id', data['annotations'])
        img_id2anns = {
            k: list(sorted(v, key=lambda a: a['id']))
            for k, v in img_id2anns.items()
        }

        categories = list(sorted(data['categories'], key=lambda a: a['id']))
        self.info.features['objects']['label'].names = [
            c['name'] for c in categories
        ]

        annotation_skipped = 0
        for image_info in sorted(data['images'], key=lambda x: x['id']):

            instances = img_id2anns.get(image_info['id'], [])

            if not instances:
                annotation_skipped += 1

            def build_bbox(x, y, width, height):
                return tfds.features.BBox(
                    ymin=max(y / image_info['height'], 0.0),
                    xmin=max(x / image_info['width'], 0.0),
                    ymax=min((y + height) / image_info['height'], 1.0),
                    xmax=min((x + width) / image_info['width'], 1.0),
                )

            example = {
                'image': os.path.join(image_dir, image_info['file_name']),
                'image/filename': image_info['file_name'],
                'image/id': image_info['id'],
                'objects': [{
                    'id': instance['id'],
                    'area': instance['area'],
                    'bbox': build_bbox(*instance['bbox']),
                    'label': instance['category_id'] - self.LABEL_OFFSET,
                    'is_crowd': bool(instance['iscrowd']),
                } for instance in instances]
            }

            yield image_info['file_name'], example
