import re
import random
from hhutil.io import fmt_path, eglob, copy

sample_ratio = 0.01
min_sample_class = 2
src = fmt_path(f"~/Downloads/datasets/Cityscapes")
tgt = fmt_path(f"~/Downloads/datasets/Cityscapes_sub")
src_image, src_label = src / "leftImg8bit_trainvaltest", src / "gtFine_trainvaltest"
tgt_image, tgt_label = tgt / "leftImg8bit_trainvaltest", tgt / "gtFine_trainvaltest"

tgt_image.mkdir(parents=True, exist_ok=True)
tgt_label.mkdir(parents=True, exist_ok=True)
for f in ["license.txt", "README"]:
    copy(src_image / f, tgt_image)
    copy(src_label / f, tgt_label)

for split in ["train", "val", "test"]:
    src_image_split, src_label_split = src_image / "leftImg8bit" / split, src_label / "gtFine" / split
    tgt_image_split, tgt_label_split = tgt_image / "leftImg8bit" / split, tgt_label / "gtFine" / split
    for class_d in eglob(src_image_split, "*"):
        class_name = class_d.stem
        print(split, class_name)
        src_image_split_class, src_label_split_class = src_image_split / class_name, src_label_split / class_name
        tgt_image_split_class, tgt_label_split_class = tgt_image_split / class_name, tgt_label_split / class_name
        tgt_image_split_class.mkdir(parents=True, exist_ok=True)
        tgt_label_split_class.mkdir(parents=True, exist_ok=True)
        image_ids = [
            re.match("([a-z]+_[0-9]{6}_[0-9]{6})_leftImg8bit", image_f.stem).group(1)
            for image_f in eglob(src_image_split_class, "*.png")
        ]
        n = len(image_ids)
        assert sum(1 for x in eglob(src_label_split_class, "*.png")) == 3 * n
        assert sum(1 for x in eglob(src_label_split_class, "*.json")) == n
        sampled_ids = random.sample(image_ids, max(int(n * sample_ratio), min_sample_class))
        for id in sampled_ids:
            copy(src_image_split_class / f"{id}_leftImg8bit.png", tgt_image_split_class)
            copy(src_label_split_class / f"{id}_gtFine_color.png", tgt_label_split_class)
            copy(src_label_split_class / f"{id}_gtFine_instanceIds.png", tgt_label_split_class)
            copy(src_label_split_class / f"{id}_gtFine_labelIds.png", tgt_label_split_class)
            copy(src_label_split_class / f"{id}_gtFine_polygons.json", tgt_label_split_class)
