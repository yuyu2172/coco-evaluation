import pickle
from eval_instance_segmentation_coco import eval_instance_segmentation_coco


with open('data/fake.pkl', 'rb') as f:
    bboxes, masks, labels, scores, keys = pickle.load(f)

with open('data/fake_gt.pkl', 'rb') as f:
    sizes, gt_bboxes, gt_masks, gt_labels, gt_crowdeds = pickle.load(f)

results = eval_instance_segmentation_coco(
    sizes, bboxes, masks, labels, scores, gt_bboxes, gt_masks, gt_labels, gt_crowdeds)

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
