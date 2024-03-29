from collections import defaultdict
from typing import List

import numpy as np
from toolz.curried import groupby

from hanser.detection.bbox import BBox


def iou_mn(boxes1, boxes2):
    boxes2 = boxes2.T
    y_min1, x_min1, y_max1, x_max1 = np.split(boxes1, 4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = np.split(boxes2, 4, axis=0)
    y_min = np.maximum(y_min1, y_min2)
    x_min = np.maximum(x_min1, x_min2)
    y_max = np.minimum(y_max1, y_max2)
    x_max = np.minimum(x_max1, x_max2)
    inter_h = np.maximum(0.0, y_max - y_min)
    inter_w = np.maximum(0.0, x_max - x_min)
    inter_area = inter_h * inter_w
    areas1 = (y_max1 - y_min1) * (x_max1 - x_min1)
    areas2 = (y_max2 - y_min2) * (x_max2 - x_min2)
    union_area = areas1 + areas2 - inter_area
    return np.where(inter_area == 0, 0.0, inter_area / union_area)



def average_precision(detections: List[BBox], ground_truths: List[BBox], iou_threshold=.5, use_07_metric=True, ignore_difficult=True):
    c2dts = groupby(lambda b: b.category_id, detections)
    c2gts = groupby(lambda b: b.category_id, ground_truths)
    dts = {}
    gts = {}
    for c, c_dts in c2dts.items():
        dts[c] = groupby(lambda b: b.image_id, c_dts)
        for i in dts[c]:
            dts[c][i].sort(key=lambda d: d.score, reverse=True)
    for c, c_gts in c2gts.items():
        gts[c] = groupby(lambda b: b.image_id, c_gts)

    img_ids = set(d.image_id for d in ground_truths)
    classes = set(d.category_id for d in ground_truths)

    ious = defaultdict(lambda: defaultdict())

    for c in classes:
        if c not in dts:
            continue
        for i in img_ids:
            if i not in dts[c] or i not in gts[c]:
                continue
            gt = gts[c][i]
            dt = dts[c][i]
            dt_bboxes = np.array([d.bbox for d in dt])
            gt_bboxes = np.array([d.bbox for d in gt])
            ious[c][i] = iou_mn(dt_bboxes, gt_bboxes)

    aps = {}
    for c in classes:
        if c not in dts:
            aps[c] = 0
            continue

        c_gts = gts[c]
        if ignore_difficult:
            n_positive = len([d for ds in c_gts.values() for d in ds if not d.is_difficult])
        else:
            n_positive = len([d for ds in c_gts.values() for d in ds])
        # n_positive = len([d for ds in c_gts.values() for d in ds if not (ignore_difficult and d.is_difficult)])
        c_dts = sorted([d for ds in dts[c].values() for d in ds], key=lambda b: b.score, reverse=True)
        TP = np.zeros(len(c_dts), dtype=np.uint8)
        FP = np.zeros(len(c_dts), dtype=np.uint8)
        seen = {
            i: np.zeros(len(ds), dtype=np.uint8)
            for i, ds in c_gts.items()
        }
        rank = {
            i: 0
            for i in c_gts
        }
        for di, dt in enumerate(c_dts):
            img_id = dt.image_id
            if img_id not in c_gts:
                FP[di] = 1
                continue
            iou = ious[c][img_id][rank[img_id]]
            rank[img_id] += 1
            j_max, iou_max = max(enumerate(iou), key=lambda x: x[1])
            if iou_max > iou_threshold:
                if not (ignore_difficult and c_gts[img_id][j_max].is_difficult):
                    if not seen[img_id][j_max]:
                        TP[di] = 1
                        seen[img_id][j_max] = 1
                    else:
                        FP[di] = 1
            else:
                FP[di] = 1
        acc_fp = np.cumsum(FP)
        acc_tp = np.cumsum(TP)
        recall = acc_tp / n_positive
        precision = acc_tp / (acc_fp + acc_tp + 1e-10)
        ap = average_precision_pr(precision, recall, use_07_metric)
        aps[c] = round(ap, 6)
    return aps


def average_precision2(detections: List[BBox], ground_truths: List[BBox], iou_threshold=.5, use_07_metric=True, ignore_difficult=True):
    c2dts = groupby(lambda b: b.category_id, detections)
    c2gts = groupby(lambda b: b.category_id, ground_truths)
    dts = {}
    gts = {}
    for c, c_dts in c2dts.items():
        dts[c] = groupby(lambda b: b.image_id, c_dts)
        for i in dts[c]:
            dts[c][i].sort(key=lambda d: d.score, reverse=True)
    for c, c_gts in c2gts.items():
        gts[c] = groupby(lambda b: b.image_id, c_gts)

    img_ids = set(d.image_id for d in ground_truths)
    classes = set(d.category_id for d in ground_truths)

    ious = defaultdict(lambda: defaultdict())

    for c in classes:
        if c not in dts:
            continue
        for i in img_ids:
            if i not in dts[c] or i not in gts[c]:
                continue
            gt = gts[c][i]
            dt = dts[c][i]
            dt_bboxes = np.array([d.bbox for d in dt])
            gt_bboxes = np.array([d.bbox for d in gt])
            ious[c][i] = iou_mn(dt_bboxes, gt_bboxes)

    aps = {}
    for c in classes:
        if c not in dts:
            aps[c] = 0
            continue

        c_gts = gts[c]
        if ignore_difficult:
            n_positive = len([d for ds in c_gts.values() for d in ds if not d.is_difficult])
        else:
            n_positive = len([d for ds in c_gts.values() for d in ds])
        # n_positive = len([d for ds in c_gts.values() for d in ds if not (ignore_difficult and d.is_difficult)])
        c_dts = sorted([d for ds in dts[c].values() for d in ds], key=lambda b: b.score, reverse=True)
        TP = np.zeros(len(c_dts), dtype=np.uint8)
        FP = np.zeros(len(c_dts), dtype=np.uint8)
        seen = {
            i: np.zeros(len(ds), dtype=np.uint8)
            for i, ds in c_gts.items()
        }
        rank = {
            i: 0
            for i in c_gts
        }
        for di, dt in enumerate(c_dts):
            img_id = dt.image_id
            if img_id not in c_gts:
                FP[di] = 1
                continue
            iou = ious[c][img_id][rank[img_id]]
            rank[img_id] += 1
            j_max, iou_max = max(enumerate(iou), key=lambda x: x[1])
            if iou_max > iou_threshold:
                if not (ignore_difficult and c_gts[img_id][j_max].is_difficult):
                    if not seen[img_id][j_max]:
                        TP[di] = 1
                        seen[img_id][j_max] = 1
                    else:
                        FP[di] = 1
            else:
                FP[di] = 1
        acc_fp = np.cumsum(FP)
        acc_tp = np.cumsum(TP)
        recall = acc_tp / n_positive
        precision = acc_tp / (acc_fp + acc_tp + 1e-10)
        ap = average_precision_pr(precision, recall, use_07_metric)
        aps[c] = round(ap, 6)
    return aps


def average_precision_pr(precision, recall, use_07_metric=True):
    if use_07_metric:
        ap = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(precision)[recall >= t])
            ap += p / 11
        return ap
    mrec = [0, *recall, 1]
    mpre = [0, *precision, 0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap += np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap