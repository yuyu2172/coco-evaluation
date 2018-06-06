import contextlib
import itertools
import numpy as np
import os
import six
import sys

try:
    import pycocotools.coco
    import pycocotools.cocoeval
    import pycocotools.mask as mask_tools
    _available = True
except ImportError:
    _available = False

from mask_utils import mask2whole_mask


def eval_instance_segmentation_coco(sizes, pred_bboxes, pred_masks,
                                    pred_labels, pred_scores,
                                    gt_bboxes, gt_masks, gt_labels,
                                    gt_crowdeds=None, gt_areas=None):
    if not _available:
        raise ValueError(
            'Please install pycocotools \n'
            'pip install -e \'git+https://github.com/pdollar/coco.git'
            '#egg=pycocotools&subdirectory=PythonAPI\'')

    gt_coco = pycocotools.coco.COCO()
    pred_coco = pycocotools.coco.COCO()

    sizes = iter(sizes)
    pred_bboxes = iter(pred_bboxes)
    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)
    gt_crowdeds = (iter(gt_crowdeds) if gt_crowdeds is not None
                   else itertools.repeat(None))
    gt_areas = (iter(gt_areas) if gt_areas is not None
                else itertools.repeat(None))

    ids = list()
    pred_annos = list()
    gt_annos = list()
    unique_labels = dict()
    for i, (size, pred_bbox, pred_mask, pred_label, pred_score,
            gt_bbox, gt_mask, gt_label, gt_crowded, gt_area) in enumerate(
                six.moves.zip(
                    sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
                    gt_bboxes, gt_masks, gt_labels, gt_crowdeds, gt_areas)):
        if gt_area is None:
            gt_area = itertools.repeat(None)
        if gt_crowded is None:
            gt_crowded = itertools.repeat(None)
        # Starting ids from 1 is important when using COCO.
        img_id = i + 1

        pred_whole_mask = mask2whole_mask(pred_mask, pred_bbox, size)
        gt_whole_mask = mask2whole_mask(gt_mask, gt_bbox, size)
        for pred_whole_m, pred_lb, pred_sc in zip(
                pred_whole_mask, pred_label, pred_score):
            pred_annos.append(
                _create_anno(pred_whole_m, pred_lb, pred_sc,
                             img_id=img_id, anno_id=len(pred_annos) + 1,
                             crw=0))
            unique_labels[pred_lb] = True

        for gt_whole_m, gt_lb, gt_crw, gt_ar in zip(
                gt_whole_mask, gt_label, gt_crowded, gt_area):
            gt_annos.append(
                _create_anno(gt_whole_m, gt_lb, None,
                             img_id=img_id, anno_id=len(gt_annos) + 1,
                             crw=gt_crw, ar=gt_ar))
            unique_labels[gt_lb] = True
        ids.append({'id': img_id, 'height': size[0], 'width': size[1]})

    pred_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    gt_coco.dataset['categories'] = [{'id': i} for i in unique_labels.keys()]
    pred_coco.dataset['annotations'] = pred_annos
    gt_coco.dataset['annotations'] = gt_annos
    pred_coco.dataset['images'] = ids
    gt_coco.dataset['images'] = ids

    with _redirect_stdout(open(os.devnull, 'w')):
        pred_coco.createIndex()
        gt_coco.createIndex()
        ev = pycocotools.cocoeval.COCOeval(gt_coco, pred_coco, 'segm')
        ev.evaluate()
        ev.accumulate()

    results = {'coco_eval': ev}
    p = ev.params
    common_kwargs = {
        'prec': ev.eval['precision'],
        'rec': ev.eval['recall'],
        'iou_threshs': p.iouThrs,
        'area_ranges': p.areaRnglb,
        'max_detection_list': p.maxDets}
    all_kwargs = {
        'ap/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.5, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.75/area=all/maxDets=100': {
            'ap': True, 'iou_thresh': 0.75, 'area_range': 'all',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ap/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': True, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=all/maxDets=1': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 1},
        'ar/iou=0.50:0.95/area=all/maxDets=10': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 10},
        'ar/iou=0.50:0.95/area=all/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'all',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=small/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'small',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=medium/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'medium',
            'max_detection': 100},
        'ar/iou=0.50:0.95/area=large/maxDets=100': {
            'ap': False, 'iou_thresh': None, 'area_range': 'large',
            'max_detection': 100},
    }

    for key, kwargs in all_kwargs.items():
        kwargs.update(common_kwargs)
        metrics, mean_metric = _summarize(**kwargs)
        results[key] = metrics
        results['m' + key] = mean_metric
    return results


def _create_anno(whole_m, lb, sc, img_id, anno_id, crw=None, ar=None):
    H, W = whole_m.shape
    if crw is None:
        crw = False
    whole_m = np.asfortranarray(whole_m.astype(np.uint8))
    rle = mask_tools.encode(whole_m)
    # Surprisingly, ground truth ar can be different from area(rle)
    if ar is None:
        ar = mask_tools.area(rle)
    anno = {
        'image_id': img_id, 'category_id': lb,
        'segmentation': rle,
        'area': ar,
        'id': anno_id,
        'iscrowd': crw}
    if sc is not None:
        anno.update({'score': sc})
    return anno


def _summarize(
        prec, rec, iou_threshs, area_ranges,
        max_detection_list,
        ap=True, iou_thresh=None, area_range='all',
        max_detection=100):
    a_idx = area_ranges.index(area_range)
    m_idx = max_detection_list.index(max_detection)
    if ap:
        val_value = prec.copy()  # (T, R, K, A, M)
        if iou_thresh is not None:
            val_value = val_value[iou_thresh == iou_threshs]
        val_value = val_value[:, :, :, a_idx, m_idx]
    else:
        val_value = rec.copy()  # (T, K, A, M)
        if iou_thresh is not None:
            val_value = val_value[iou_thresh == iou_threshs]
        val_value = val_value[:, :, a_idx, m_idx]

    val_value[val_value == -1] = np.nan
    val_value = val_value.reshape((-1, val_value.shape[-1]))
    valid_classes = np.any(np.logical_not(np.isnan(val_value)), axis=0)
    cls_val_value = np.nan * np.ones(len(valid_classes), dtype=np.float32)
    cls_val_value[valid_classes] = np.nanmean(
        val_value[:, valid_classes], axis=0)

    if not np.any(valid_classes):
        mean_val_value = np.nan
    else:
        mean_val_value = np.nanmean(cls_val_value)
    return cls_val_value, mean_val_value


@contextlib.contextmanager
def _redirect_stdout(target):
    original = sys.stdout
    sys.stdout = target
    yield
    sys.stdout = original
