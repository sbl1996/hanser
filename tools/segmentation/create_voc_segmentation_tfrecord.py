import argparse
from hhutil.io import fmt_path
from hanser.datasets.segmentation.tfrecord import convert_segmentation_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert VOC Segmentation to TFRecord')
    parser.add_argument('-r', '--root', help='VOC root')
    parser.add_argument('-o', '--output_dir', help='output path')
    parser.add_argument('-p', '--part', help='part')
    parser.add_argument('-n', '--num_shards', help='number of shards')
    args = parser.parse_args()

    root = fmt_path(args.root)
    output_dir = fmt_path(args.output_dir)
    part = args.part
    num_shards = int(args.num_shards)

    split_f = root / f"ImageSets/Segmentation/{part}.txt"
    image_dir = root / 'JPEGImages'
    label_dir = root / root / 'SegmentationClassAug'

    convert_segmentation_dataset(split_f, output_dir, image_dir, label_dir, num_shards=num_shards)