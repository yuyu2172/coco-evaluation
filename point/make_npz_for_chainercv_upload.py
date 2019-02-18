import pickle
import numpy as np


with open('data/fake.pkl', 'rb') as f:
    points, labels, scores, keys = pickle.load(f)

with open('data/fake_gt.pkl', 'rb') as f:
    gt_points, gt_bboxes, gt_labels, gt_areas, gt_crowdeds = pickle.load(f)

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

np.savez('eval_point_coco_dataset_2019_02_18.npz',
         points=gt_points_yx,
         is_valids=gt_point_is_valids,
         bboxes=gt_bboxes,
         labels=gt_labels,
         areas=gt_areas,
         crowdeds=gt_crowdeds)
np.savez('eval_point_coco_result_2019_02_18.npz',
         points=points_yx,
         scores=scores,
         labels=labels,)
