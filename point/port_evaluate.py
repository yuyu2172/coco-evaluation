import pickle
from eval_point_coco import eval_point_coco
import pycocotools.cocoeval


with open('data/fake.pkl', 'rb') as f:
    points, labels, scores, keys = pickle.load(f)

with open('data/fake_gt.pkl', 'rb') as f:
    gt_points, gt_bboxes, gt_labels, gt_areas, gt_crowdeds = pickle.load(f)

# for gt_point in gt_points:
#     for pnt in gt_point:
#         pnt[pnt[:, 2] > 0, 2] = 1
gt_points_yx = []
gt_point_is_valids = []
for gt_point in gt_points:
    gt_point_yx = []
    gt_point_is_valid = []
    for pnt in gt_point:
        gt_point_yx.append(pnt[:, :2])
        gt_point_is_valid.append(pnt[:, 2])
    gt_points_yx.append(gt_point_yx)
    gt_point_is_valids.append(gt_point_is_valid)

points_yx = []
for point in points:
    point_yx = []
    for pnt in point:
        point_yx.append(pnt[:, :2])
    points_yx.append(point_yx)
results = eval_point_coco(
    points_yx, labels, scores,
    gt_points_yx, gt_point_is_valids,
    gt_bboxes, gt_labels, gt_areas, gt_crowdeds)


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
