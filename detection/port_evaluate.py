import pickle
from eval_detection_coco import eval_detection_coco
import pycocotools.cocoeval


with open('data/fake.pkl', 'rb') as f:
    bboxes, labels, scores, keys = pickle.load(f)

with open('data/fake_gt.pkl', 'rb') as f:
    gt_bboxes, gt_labels, gt_crowdeds, gt_areas = pickle.load(f)

results = eval_detection_coco(bboxes, labels, scores, gt_bboxes, gt_labels, gt_crowdeds, gt_areas)


keys = ['ap/iou=0.50:0.95/area=all/maxDets=100',
        'ap/iou=0.50/area=all/maxDets=100',
        'ap/iou=0.75/area=all/maxDets=100',
        'ap/iou=0.50:0.95/area=small/maxDets=100',
        'ap/iou=0.50:0.95/area=medium/maxDets=100',
        'ap/iou=0.50:0.95/area=large/maxDets=100',
        'ar/iou=0.50:0.95/area=all/maxDets=1',
        'ar/iou=0.50:0.95/area=all/maxDets=10',
        'ar/iou=0.50:0.95/area=all/maxDets=100',
        'ar/iou=0.50:0.95/area=small/maxDets=100',
        'ar/iou=0.50:0.95/area=medium/maxDets=100',
        'ar/iou=0.50:0.95/area=large/maxDets=100',
        ]

for key in keys:
    print('m' + key, results['m' + key])
