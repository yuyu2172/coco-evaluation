# COCO evaluation (instance segmentation)

This is a collection of scripts I use to port COCO evaluation code.

## Setup
1. Move `instances_val2014.json` under `data/`. The JSON can be downloaded from the official page.
2. Move `instances_val2014_fakesegm100_results.json` udner `data/`.
The json can be found here https://github.com/cocodataset/cocoapi/tree/master/results 


## What it can do

```bash
# This parses annotations in fakebbox100 into formats used in ChainerCV
$ python parse_json.py
# Run port code
$ python port_evaluate.py

# Run original evaluation
$ python original_evaluate.py
```


## Results

```bash
$ python original_evaluate.py                                                           
Running demo for *segm* results.
loading annotations into memory...
Done (t=5.21s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.28s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.310
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.381
```

The more precise scores are the following

```
[ 0.32170935  0.56469292  0.30133106  0.38737403  0.31018272  0.32693391   
  0.27037258  0.41759154  0.41898236  0.46944986  0.37675923  0.38147151]  
```


```bash
$ python port_evaluate.py                                               
('map/iou=0.50:0.95/area=all/maxDets=100', 0.32170936)          
('map/iou=0.50/area=all/maxDets=100', 0.56469297)               
('map/iou=0.75/area=all/maxDets=100', 0.30133104)               
('map/iou=0.50:0.95/area=small/maxDets=100', 0.38737401)        
('map/iou=0.50:0.95/area=medium/maxDets=100', 0.31018272)       
('map/iou=0.50:0.95/area=large/maxDets=100', 0.32693389)        
('mar/iou=0.50:0.95/area=all/maxDets=1', 0.27037254)            
('mar/iou=0.50:0.95/area=all/maxDets=10', 0.41759151)           
('mar/iou=0.50:0.95/area=all/maxDets=100', 0.41898233)          
('mar/iou=0.50:0.95/area=small/maxDets=100', 0.46944991)        
('mar/iou=0.50:0.95/area=medium/maxDets=100', 0.3767592)        
('mar/iou=0.50:0.95/area=large/maxDets=100', 0.38147154)        
```
