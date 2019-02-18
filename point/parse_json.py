import json
import numpy as np
import pickle

from chainercv.datasets import COCOPointDataset
from chainercv.datasets import coco_point_names
from chainercv.datasets import coco_bbox_label_names

with open('data/categories.pkl', 'rb') as f:
    categories = pickle.load(f)
cat_map = dict()
for cat in categories:
    cat_map[cat['id']] = cat['name']

anns = json.load(open('data/person_keypoints_val2014_fakekeypoints100_results.json'))

label_map = dict()
data = dict()
count = 0
for ann in anns:
    if ann['image_id'] not in data:
        data[ann['image_id']] = list()

    point = np.array(ann['keypoints']).reshape(-1, 3)
    point = point[:, [1, 0, 2]]

    data[ann['image_id']].append(
        {'point': point,
         'score': ann['score'],
         'label': coco_bbox_label_names.index(cat_map[ann['category_id']])
         })


points = list()
labels = list()
scores = list()
keys = sorted(data.keys())
print(keys)
for key in keys:
    data_i = data[key]
    points.append(np.array([d['point'] for d in data_i], dtype=np.float32))
    labels.append(np.array([d['label'] for d in data_i], dtype=np.int32))
    scores.append(np.array([d['score'] for d in data_i], dtype=np.float32))

with open('data/fake.pkl', 'wb') as f:
    pickle.dump((points, labels, scores, keys), f)


##############################################################################
# dataset = COCOBboxDataset(split='val', return_crowded=True, return_area=True, use_crowded=True)
dataset = COCOPointDataset(
    year='2014',
    split='val', return_crowded=True, return_area=True, use_crowded=True)
indices = [dataset.ids.index(key) for key in keys]
gts = dataset[indices]
gt_points = [gt[1] for gt in gts]
gt_bboxes = [gt[2] for gt in gts]
gt_labels = [gt[3] for gt in gts]
gt_areas = [gt[4] for gt in gts]
gt_crowded = [gt[5] for gt in gts]

with open('data/fake_gt.pkl', 'wb') as f:
    pickle.dump((gt_points, gt_bboxes, gt_labels, gt_areas, gt_crowded), f) 

# import matplotlib.pyplot as plt
# from chainercv.visualizations import vis_bbox
# 
# img, bbox, label, crowded, _ = dataset[indices[12]]
# vis_bbox(img, bbox)
# plt.show()
# print crowded
