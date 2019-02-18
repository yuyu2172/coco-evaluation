from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here

#initialize COCO ground truth api
annFile = 'data/person_keypoints_val2014.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='data/person_keypoints_val2014_fakekeypoints100_results.json'
cocoDt=cocoGt.loadRes(resFile)


imgIds = [136, 139, 192, 241, 257,
          294, 328, 338, 395, 397,
          415, 428, 459, 474, 488,
          536, 544, 564, 569, 589,
          692, 693, 761, 764, 785,
          810, 831, 836, 872, 885,
          962, 969, 974, 985, 999,
          1000, 1089, 1146, 1149, 1164,
          1176, 1180, 1244, 1268, 1270,
          1290, 1292]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
# cocoEval.params.areaRng = cocoEval.params.areaRng[:1]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
