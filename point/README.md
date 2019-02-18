# COCO evaluation

This is a collection of scripts I use to port COCO evaluation code.

## Setup

1. Download  annotation. http://images.cocodataset.org/annotations/annotations_trainval2014.zip
2. `wget https://raw.githubusercontent.com/cocodataset/cocoapi/master/results/person_keypoints_val2014_fakekeypoints100_results.json; mkdir data; mv person_keypoints_val2014_fakekeypoints100_results.json data;`

## What it can do

```bash
# This parses annotations in fakebbox100 into formats used in ChainerCV
$ python parse_json.py
# Run port code
$ python port_evaluate.py

# Run original evaluation
$ python original_evaluate.py
```


```
$ python original_evaluate.py
loading annotations into memory...                                                       
Done (t=1.55s)                                                                           
creating index...                                                                        
index created!                                                                           
Loading and preparing results...                                              
DONE (t=0.01s)                                                                           
creating index...                                                                        
index created!                                                                           
Running per image evaluation...                                                          
Evaluate annotation type *keypoints*                                                     
DONE (t=0.06s).                                                                          
Accumulating evaluation results...                                                       
DONE (t=0.00s).                                                                          
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.377          
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.645          
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.355          
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.389          
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.392          
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.522          
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.745          
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.511          
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.515          
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.530          
```

```bash
$ python port_evaluate.py                                               

map/iou=0.50:0.95/area=all/max_dets=20 0.37733573
map/iou=0.50/area=all/max_dets=20 0.64488417
map/iou=0.75/area=all/max_dets=20 0.3546909
map/iou=0.50:0.95/area=medium/max_dets=20 0.3894106
map/iou=0.50:0.95/area=large/max_dets=20 0.39169297
mar/iou=0.50:0.95/area=all/max_dets=20 0.5218978
mar/iou=0.50/area=all/max_dets=20 0.74452555
mar/iou=0.75/area=all/max_dets=20 0.5109489
mar/iou=0.50:0.95/area=medium/max_dets=20 0.5150685
mar/iou=0.50:0.95/area=large/max_dets=20 0.5296875
```
