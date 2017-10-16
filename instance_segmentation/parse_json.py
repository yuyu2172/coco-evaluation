import json
import numpy as np
import pickle
from fcis.datasets.coco_utils import coco_instance_segmentation_label_names
# from fcis.datasets.coco_instance_segmentation_dataset import COCOInstanceSegmentationDataset
from coco_instance_segmentation_dataset import COCOInstanceSegmentationDataset

from pycocotools.mask import decode
from pycocotools.mask import encode
from pycocotools.mask import toBbox

from mask_utils import whole_mask2mask


with open('data/categories.pkl', 'rb') as f:
    categories = pickle.load(f)
cat_map = dict()
for cat in categories:
    cat_map[cat['id']] = cat['name']

anns = json.load(open('data/instances_val2014_fakesegm100_results.json'))

label_map = dict()
data = dict()
count = 0
for ann in anns:
    if ann['image_id'] not in data:
        data[ann['image_id']] = list()

    whole_m = decode(ann['segmentation'])
    # assert encode(whole_m)['counts'] == ann['segmentation']['counts']
    x_min, y_min, width, height = toBbox(ann['segmentation'])
    y_max = y_min + height
    x_max = x_min + width
    data[ann['image_id']].append(
        {'bbox': [y_min, x_min, y_max, x_max],
         'score': ann['score'],
         'whole_mask': whole_m,
         'label': coco_instance_segmentation_label_names.index(cat_map[ann['category_id']])
         })

bboxes = list()
labels = list()
scores = list()
masks = list()
keys = sorted(data.keys())
print(keys)
for key in keys:
    data_i = data[key]
    bboxes.append(np.array([d['bbox'] for d in data_i], dtype=np.float32))
    labels.append(np.array([d['label'] for d in data_i], dtype=np.int32))
    scores.append(np.array([d['score'] for d in data_i], dtype=np.float32))
    whole_mask = np.array([d['whole_mask'] for d in data_i], dtype=np.bool)
    masks.append(whole_mask2mask(whole_mask, bboxes[-1]))
    # masks.append(whole_mask)

data = {'bboxes': bboxes, 'masks': masks, 'labels': labels, 'scores': scores, 'keys': keys}
np.savez('data/coco_instance_segm_result_val2014_fakesegm100.npz', **data)


##############################################################################
dataset = COCOInstanceSegmentationDataset(
    split='val', return_crowded=True, use_crowded=True, return_area=True)
indices = [dataset.ids.index(key) for key in keys]
gts = dataset[indices]
sizes = [gt[0].shape[1:] for gt in gts]
gt_bboxes = [gt[1] for gt in gts]
gt_labels = [gt[2] for gt in gts]
gt_masks = [gt[3] for gt in gts]
gt_crowdeds = [gt[4] for gt in gts]
gt_areas = [gt[5] for gt in gts]

data = {'sizes': sizes, 'bboxes': gt_bboxes, 'masks': gt_masks, 'labels': gt_labels, 'crowdeds': gt_crowdeds, 'areas': gt_areas}
np.savez('data/coco_instance_segm_dataset_val2014_fakesegm100.npz', **data)
