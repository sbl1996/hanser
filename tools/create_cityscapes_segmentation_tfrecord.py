import re
import argparse
from hhutil.io import fmt_path, eglob, copy
from hanser.datasets.segmentation.tfrecord import convert_segmentation_dataset

def create_split_file(image_dir, target):
    image_ids = [
        re.match("([a-z]+_[0-9]{6}_[0-9]{6})_leftImg8bit", f.stem).group(1)
        for f in eglob(image_dir, "*/*.png")
    ]
    split_f = fmt_path(target)
    split_f.write_text('\n'.join(image_ids))
    return split_f

def image_stem_transform(file_name):
    class_name = file_name[:-14]
    return f"{class_name}/{file_name}_leftImg8bit"

def label_stem_transform(file_name):
    class_name = file_name[:-14]
    return f"{class_name}/{file_name}_gtFine_labelIds"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert VOC Segmentation to TFRecord')
    parser.add_argument('-r', '--root', help='VOC root')
    parser.add_argument('-o', '--output_dir', help='output path')
    parser.add_argument('-s', '--split', help='split')
    parser.add_argument('-n', '--num_shards', help='number of shards')
    args = parser.parse_args()

    root = fmt_path(args.root)
    output_dir = fmt_path(args.output_dir)
    split = args.split
    num_shards = int(args.num_shards)

    image_dir = root / f'leftImg8bit_trainvaltest/leftImg8bit/{split}'
    label_dir = root / f"gtFine_trainvaltest/gtFine/{split}"
    split_f = root / f"{split}.txt"
    create_split_file(image_dir, split_f)

    convert_segmentation_dataset(
        split_f, output_dir, image_dir, label_dir,
        image_format="png", label_format="png", num_shards=num_shards,
        image_stem_transform=image_stem_transform,
        label_stem_transform=label_stem_transform
    )