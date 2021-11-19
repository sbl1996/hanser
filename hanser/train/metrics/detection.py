from toolz import get

import numpy as np

import tensorflow as tf

from hanser.detection.bbox import BBox
from hanser.detection.eval import average_precision
from hhutil.io import download_file, fmt_path
from hhutil.hash import md5


def download_instances_val2017(save_dir="./"):
    save_dir = fmt_path(save_dir)
    fp = save_dir / "instances_val2017.json"
    if fp.exists() and md5(fp) == "b681580a54b900b3cb44022fd1102ad5":
        return fp
    else:
        url = "https://github.com/sbl1996/hanser/releases/download/0.1.3/instances_val2017.json"
        return download_file(url, save_dir)


class MeanAveragePrecision:

    def __init__(self, iou_threshold=0.5, interpolation='11point', ignore_difficult=True, class_names=None, output_transform=None):

        self.iou_threshold = iou_threshold
        self.interpolation = interpolation
        self.ignore_difficult = ignore_difficult
        self.class_names = class_names
        self.output_transform = output_transform

        self.gts = []
        self.dts = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.output_transform is not None:
            y_pred = self.output_transform(y_pred)

        all_dt_bboxes, all_dt_classes, all_dt_scores, all_dt_n_valids = [
            t.numpy() for t in get(['bbox', 'label', 'score', 'n_valid'], y_pred)
        ]

        all_gt_bboxes, all_gt_labels, image_ids = [
            t.numpy() for t in get(['gt_bbox', 'gt_label', 'image_id'], y_true)
        ]
        if 'is_difficult' in y_true:
            all_is_difficults = y_true['is_difficult'].numpy()
        else:
            all_is_difficults = np.full_like(all_gt_labels, False, dtype=np.bool)

        all_gt_n_valids = np.sum(all_gt_labels != 0, axis=1)
        all_gt_labels -= 1
        batch_size, num_dets = all_dt_bboxes.shape[:2]
        for i in range(batch_size):
            image_id = image_ids[i]
            for j in range(all_dt_n_valids[i]):
                self.dts.append({
                    'image_id': image_id,
                    'bbox': all_dt_bboxes[i, j],
                    'category_id': all_dt_classes[i, j],
                    'score': all_dt_scores[i, j],
                })

            for j in range(all_gt_n_valids[i]):
                self.gts.append({
                    'image_id': image_id,
                    'bbox': all_gt_bboxes[i, j],
                    'category_id': all_gt_labels[i, j],
                    'is_difficult': all_is_difficults[i, j]
                })

    def result(self):
        dts = [BBox(**ann) for ann in self.dts]
        gts = [BBox(**ann) for ann in self.gts]

        aps = average_precision(dts, gts, self.iou_threshold, self.interpolation == '11point', self.ignore_difficult)
        mAP = np.mean(list(aps.values()))
        if self.class_names:
            num_classes = len(self.class_names)
            d = {}
            for i in range(num_classes):
                d[self.class_names[i]] = aps.get(i, 0) * 100
            d['ALL'] = mAP * 100
            # d = pd.DataFrame({'mAP': d}).transpose()
            # pd.set_option('precision', 1)
            print(d)
        return tf.convert_to_tensor(mAP)

    def reset_states(self):
        self.gts = []
        self.dts = []


def _bbox_transform(bbox, output_size, image_size):
    ow, oh = output_size
    iw, ih = image_size
    scale = min(ow / iw, oh / ih)
    w, h = int(iw * scale), int(ih * scale)
    # pw, ph = ow - w, oh - h
    bx, by, bw, bh = bbox
    bbox = [
        bx / w * iw,
        by / h * ih,
        bw / w * iw,
        bh / h * ih,
    ]
    return bbox



class COCOEval:

    def __init__(self, ann_file, output_size, output_transform=None,
                 bbox_transform=_bbox_transform, label_transform=lambda x: x+1):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.output_size = output_size
        self.output_transform = output_transform
        self.bbox_transform = bbox_transform
        self.label_transform = label_transform

        self.image_ids = []
        self.dts = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.output_transform is not None:
            y_pred = self.output_transform(y_pred)

        all_dt_bboxes, all_dt_classes, all_dt_scores, all_dt_n_valids = [
            t.numpy() for t in get(['bbox', 'label', 'score', 'n_valid'], y_pred)
        ]

        image_ids = y_true['image_id'].numpy()

        batch_size, num_dets = all_dt_bboxes.shape[:2]
        for i in range(batch_size):
            image_id = int(image_ids[i])
            info = self.coco.loadImgs(image_id)[0]
            width, height = info['width'], info['height']
            for j in range(all_dt_n_valids[i]):
                bbox = all_dt_bboxes[i, j]
                x, y = bbox[1], bbox[0]
                w, h = bbox[3] - bbox[1], bbox[2] - bbox[0]
                bbox = [x, y, w, h]
                bbox = self.bbox_transform(bbox, self.output_size, (width, height))
                label = self.label_transform(all_dt_classes[i, j])
                self.dts.append({
                    'image_id': image_id,
                    'bbox': bbox,
                    'category_id': label,
                    'score': all_dt_scores[i, j],
                })

            self.image_ids.append(image_id)

    def result(self):
        from pycocotools.cocoeval import COCOeval as _COCOeval
        coco_dt = self.coco.loadRes(self.dts)
        E = _COCOeval(self.coco, coco_dt, iouType='bbox')
        self.E = E
        E.params.imgIds = sorted(self.image_ids)
        E.evaluate()
        E.accumulate()
        E.summarize()
        return tf.convert_to_tensor(E.stats[0])

    def reset_states(self):
        self.image_ids = []
        self.dts = []
