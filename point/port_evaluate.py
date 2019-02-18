import pickle
from eval_point_coco import eval_point_coco
import pycocotools.cocoeval


with open('data/fake.pkl', 'rb') as f:
    points, labels, scores, keys = pickle.load(f)

with open('data/fake_gt.pkl', 'rb') as f:
    gt_points, gt_bboxes, gt_labels, gt_areas, gt_crowdeds = pickle.load(f)

results = eval_point_coco(
    points, labels, scores, gt_points, gt_bboxes, gt_labels, gt_areas, gt_crowdeds)


keys = ['map/iou=0.50:0.95/area=all/max_dets=20',
        'map/iou=0.50/area=all/max_dets=20',
        'map/iou=0.75/area=all/max_dets=20',
        'map/iou=0.50:0.95/area=medium/max_dets=20',
        'map/iou=0.50:0.95/area=large/max_dets=20',
        'mar/iou=0.50:0.95/area=all/max_dets=20',
        'mar/iou=0.50/area=all/max_dets=20',
        'mar/iou=0.75/area=all/max_dets=20',
        'mar/iou=0.50:0.95/area=medium/max_dets=20',
        'mar/iou=0.50:0.95/area=large/max_dets=20',
        ]

print()
for key in keys:
    print(key, results[key])
