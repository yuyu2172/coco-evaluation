import json
import numpy as np
import pickle

from chainercv.datasets import COCOBboxDataset
from chainercv.datasets import coco_bbox_label_names

with open('data/categories.pkl', 'rb') as f:
    categories = pickle.load(f)
cat_map = dict()
for cat in categories:
    cat_map[cat['id']] = cat['name']

anns = json.load(open('data/instances_val2014_fakebbox100_results.json'))

label_map = dict()
data = dict()
count = 0
for ann in anns:
    if ann['image_id'] not in data:
        data[ann['image_id']] = list()

    x_min, y_min, width, height = ann['bbox']
    y_max = y_min + height
    x_max = x_min + width

    data[ann['image_id']].append(
        {'bbox': [y_min, x_min, y_max, x_max],
         'score': ann['score'],
         'label': coco_bbox_label_names.index(cat_map[ann['category_id']]) + 1
         })


bboxes = list()
labels = list()
scores = list()
keys = sorted(data.keys())
print keys
for key in keys:
    data_i = data[key]
    bboxes.append(np.array([d['bbox'] for d in data_i], dtype=np.float32))
    labels.append(np.array([d['label'] for d in data_i], dtype=np.int32))
    scores.append(np.array([d['score'] for d in data_i], dtype=np.float32))

with open('data/fake.pkl', 'wb') as f:
    pickle.dump((bboxes, labels, scores, keys), f)


##############################################################################
dataset = COCOBboxDataset(split='val', return_crowded=True, return_area=True, use_crowded=True)
indices = [dataset.ids.index(key) for key in keys]
gts = dataset[indices]
gt_bboxes = [gt[1] for gt in gts]
gt_labels = [gt[2] + 1 for gt in gts]
gt_crowded = [gt[3] for gt in gts]
gt_areas = [gt[4] for gt in gts]

with open('data/fake_gt.pkl', 'wb') as f:
    pickle.dump((gt_bboxes, gt_labels, gt_crowded, gt_areas), f) 

import matplotlib.pyplot as plt
from chainercv.visualizations import vis_bbox

img, bbox, label, crowded, _ = dataset[indices[12]]
vis_bbox(img, bbox)
plt.show()
print crowded
