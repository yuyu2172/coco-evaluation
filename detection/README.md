# COCO evaluation

This is a collection of scripts I use to port COCO evaluation code.

## Setup
1. Move `instances_val2014.json` under `data/`. The JSON can be downloaded from the official page.
2. Move `instances_val2014_fakebbox100_results.json` udner `data/`.
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
Running demo for *bbox* results.                                                   
loading annotations into memory...                                                 
Done (t=5.31s)                                                                     
creating index...                                                                  
index created!                                                                     
Loading and preparing results...                                                   
DONE (t=0.02s)                                                                     
creating index...                                                                  
index created!                                                                     
Running per image evaluation...                                                    
Evaluate annotation type *bbox*                                                    
DONE (t=0.24s).                                                                    
Accumulating evaluation results...                                                 
DONE (t=0.20s).                                                                    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.507    
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.699    
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.575    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.586    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519    
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.389    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.596    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.598    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.640    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566    
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564    
```


```bash
$ python port_evaluate.py                                               
('map/iou=0.50:0.95/area=all/maxDets=100', 0.50698525)
('map/iou=0.50/area=all/maxDets=100', 0.69937724)
('map/iou=0.75/area=all/maxDets=100', 0.57538623)
('map/iou=0.50:0.95/area=small/maxDets=100', 0.58562577)
('map/iou=0.50:0.95/area=medium/maxDets=100', 0.5193997)
('map/iou=0.50:0.95/area=large/maxDets=100', 0.50139791)
('mar/iou=0.50:0.95/area=all/maxDets=1', 0.38919371)
('mar/iou=0.50:0.95/area=all/maxDets=10', 0.59606051)
('mar/iou=0.50:0.95/area=all/maxDets=100', 0.59773397)
('mar/iou=0.50:0.95/area=small/maxDets=100', 0.63981092)
('mar/iou=0.50:0.95/area=medium/maxDets=100', 0.56642061)
('mar/iou=0.50:0.95/area=large/maxDets=100', 0.56429064)
```


