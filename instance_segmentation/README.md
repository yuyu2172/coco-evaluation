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


```bash
$ python port_evaluate.py                                               
('map/iou=0.50:0.95/area=all/maxDets=100', 0.31732336)     
('map/iou=0.50/area=all/maxDets=100', 0.56643742)          
('map/iou=0.75/area=all/maxDets=100', 0.29026759)          
('map/iou=0.50:0.95/area=small/maxDets=100', 0.37689963)   
('map/iou=0.50:0.95/area=medium/maxDets=100', 0.30473712)  
('map/iou=0.50:0.95/area=large/maxDets=100', 0.327869)     
('mar/iou=0.50:0.95/area=all/maxDets=1', 0.26770136)       
('mar/iou=0.50:0.95/area=all/maxDets=10', 0.4126409)       
('mar/iou=0.50:0.95/area=all/maxDets=100', 0.41403174)     
('mar/iou=0.50:0.95/area=small/maxDets=100', 0.45822001)   
('mar/iou=0.50:0.95/area=medium/maxDets=100', 0.36902145)  
('mar/iou=0.50:0.95/area=large/maxDets=100', 0.3824715)    

```


The two results are slightly different (maximum of 0.01).
The discrepancy may have arised from noisy introduced from the ways in which masks are handled.
In our impelmentation, a mask in RLE format is converted into numpy array (see dataset class) and then put back into RLE (see evaluation scipt).
On the other hand, the original implementation uses RLE directly for evaluation.
If the `RLE->array->RLE` does not recover the original `RLE`, the value may change.
