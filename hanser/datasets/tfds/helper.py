import dataclasses
import tensorflow_datasets as tfds

class DatasetInfo(tfds.core.DatasetInfo):

    def initialize_from_bucket(self):
        return


class HGeneratorBasedBuilder(tfds.core.GeneratorBasedBuilder):

    def download_and_prepare(self, *, download_dir=None, download_config=None):
        download_config = dataclasses.replace(download_config, try_download_gcs=False)
        return super().download_and_prepare(download_dir=download_dir, download_config=download_config)


def load(name: str, *, split, data_dir=None, shuffle_files: bool = False):
    return tfds.load(
        name, split=split, data_dir=data_dir, download=False,
        shuffle_files=shuffle_files, read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True))
