import os
import json
import collections
import tensorflow as tf
import tensorflow_datasets as tfds


class GenDet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for svhn dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """Put it anywhere you like."""

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = {
            # Images can have variable shape
            'image': tfds.features.Image(encoding_format='png'),
            'image/filename': tfds.features.Text(),
            'image/id': tf.int32,
            'objects': tfds.features.Sequence({
                'id': tf.int64,
                'area': tf.int64,
                'bbox': tfds.features.BBoxFeature(),
                'label': tfds.features.ClassLabel(num_classes=10),
                'is_crowd': tf.bool,
            }),
        }

        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(features),
        )

    def _split_generators(self, dl_manager):
        train_image_dir = os.path.join(dl_manager.manual_dir, 'train')
        train_ann_file = os.path.join(dl_manager.manual_dir, 'train.json')

        val_image_dir = os.path.join(dl_manager.manual_dir, 'val')
        val_ann_file = os.path.join(dl_manager.manual_dir, 'val.json')

        test_image_dir = os.path.join(dl_manager.manual_dir, 'test')
        test_ann_file = os.path.join(dl_manager.manual_dir, 'test.json')

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    image_dir=train_image_dir,
                    ann_file=train_ann_file,
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(
                    image_dir=val_image_dir,
                    ann_file=val_ann_file,
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    image_dir=test_image_dir,
                    ann_file=test_ann_file,
                )),
        ]

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
        img_id2anns = collections.defaultdict(list)
        for a in data['annotations']:
            img_id2anns[a['image_id']].append(a)
        self._img_id2anns = {
            k: list(sorted(v, key=lambda a: a['id']))
            for k, v in img_id2anns.items()
        }

        categories = data['categories']
        images = data['images']

        objects_key = 'objects'
        self.info.features[objects_key]['label'].names = [
            c['name'] for c in categories
        ]

        # Iterate over all images
        annotation_skipped = 0
        for image_info in sorted(images, key=lambda x: x['id']):

            instances = self._img_id2anns[image_info['id']]

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
                objects_key: [{
                    'id': instance['id'],
                    'area': instance['area'],
                    'bbox': build_bbox(*instance['bbox']),
                    'label': instance['category_id'],
                    'is_crowd': bool(instance['iscrowd']),
                } for instance in instances]
            }

            yield image_info['file_name'], example
