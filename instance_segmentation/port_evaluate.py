import numpy as np
from eval_instance_segmentation_coco import eval_instance_segmentation_coco


# with open('data/fake.pkl', 'rb') as f:
#     bboxes, masks, labels, scores, keys = pickle.load(f)
# 
# with open('data/fake_gt.pkl', 'rb') as f:
#     sizes, gt_bboxes, gt_masks, gt_labels, gt_crowdeds, gt_areas = pickle.load(f)
pred = np.load('data/coco_instance_segm_result_val2014_fakesegm100.npz')
bboxes = pred['bboxes']
masks = pred['masks']
labels = pred['labels']
scores = pred['scores']
gt = np.load('data/coco_instance_segm_dataset_val2014_fakesegm100.npz')
sizes = gt['sizes']
gt_bboxes = gt['bboxes']
gt_masks = gt['masks']
gt_labels = gt['labels']
gt_crowdeds = gt['crowdeds']
gt_areas = gt['areas']

results = eval_instance_segmentation_coco(
    sizes, bboxes, masks, labels, scores, gt_bboxes, gt_masks, gt_labels, gt_crowdeds, gt_areas)

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
