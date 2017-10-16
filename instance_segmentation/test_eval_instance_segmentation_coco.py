import numpy as np
import os
from six.moves.urllib import request
import unittest

from chainer import testing

from eval_instance_segmentation_coco import eval_instance_segmentation_coco

try:
    import pycocotools  # NOQA
    optional_modules = True
except ImportError:
    optional_modules = False


data = {
    'pred_bboxes': [
        [[0, 0, 2, 2]]],
    'pred_masks': [
        [[[True, True], [True, True]]]],
    'pred_labels': [
        [0]],
    'pred_scores': [
        [0.8]],
    'gt_bboxes': [
        [[0, 0, 2, 2]]],
    'gt_masks': [
        [[[True, True], [True, True]]]],
    'gt_labels': [
        [0]]}


class TestEvalInstanceSegmentationCOCOSimple(unittest.TestCase):

    def setUp(self):
        self.pred_bboxes = (np.array(bbox) for bbox in data['pred_bboxes'])
        self.pred_masks = (np.array(mask) for mask in data['pred_masks'])
        self.pred_labels = (np.array(label) for label in data['pred_labels'])
        self.pred_scores = (np.array(score) for score in data['pred_scores'])
        self.gt_bboxes = (np.array(bbox) for bbox in data['gt_bboxes'])
        self.gt_masks = (np.array(mask) for mask in data['gt_masks'])
        self.gt_labels = (np.array(label) for label in data['gt_labels'])
        self.sizes = ([(10, 10)])

    def test_crowded(self):
        if not optional_modules:
            return
        result = eval_instance_segmentation_coco(
            self.sizes,
            self.pred_bboxes, self.pred_masks, self.pred_labels,
            self.pred_scores,
            self.gt_bboxes, self.gt_masks, self.gt_labels,
            gt_crowdeds=[[True]])
        # When the only ground truth is crowded, nothing is evaluated.
        # In that case, all the results are nan.
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))

    def test_area_default(self):
        if not optional_modules:
            return
        result = eval_instance_segmentation_coco(
            self.sizes,
            self.pred_bboxes, self.pred_masks, self.pred_labels,
            self.pred_scores,
            self.gt_bboxes, self.gt_masks, self.gt_labels)
        # Test that the original mask area is used, which is 4.
        # In that case, the ground truth mask is assigned to segment
        # "small".
        # Therefore, the score for segments "medium" and "large" will be nan.
        self.assertFalse(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))

    def test_area_specified(self):
        if not optional_modules:
            return
        result = eval_instance_segmentation_coco(
            self.sizes,
            self.pred_bboxes, self.pred_masks, self.pred_labels,
            self.pred_scores,
            self.gt_bboxes, self.gt_masks, self.gt_labels,
            gt_areas=[[2048]],
        )
        self.assertFalse(
            np.isnan(result['map/iou=0.50:0.95/area=medium/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=small/maxDets=100']))
        self.assertTrue(
            np.isnan(result['map/iou=0.50:0.95/area=large/maxDets=100']))


class TestEvalInstanceSegmentationCOCO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base_url = 'https://github.com/yuyu2172/' \
            'coco-evaluation/releases/download/0.0.5'

        cls.dataset = np.load('test_data/coco_instance_segm_dataset_val2014_fakesegm100.npz')
        cls.result = np.load('test_data/coco_instance_segm_result_val2014_fakesegm100.npz')

    def test_eval_detection_voc(self):
        if not optional_modules:
            return
        pred_bboxes = self.result['bboxes']
        pred_masks = self.result['masks']
        pred_labels = self.result['labels']
        pred_scores = self.result['scores']

        sizes = self.dataset['sizes']
        gt_bboxes = self.dataset['bboxes']
        gt_masks = self.dataset['masks']
        gt_labels = self.dataset['labels']
        gt_crowdeds = self.dataset['crowdeds']
        gt_areas = self.dataset['areas']

        result = eval_instance_segmentation_coco(
            sizes, pred_bboxes, pred_masks, pred_labels, pred_scores,
            gt_bboxes, gt_masks, gt_labels, gt_crowdeds, gt_areas)

        expected = {
            'map/iou=0.50:0.95/area=all/maxDets=100': 0.32170935,
            'map/iou=0.50/area=all/maxDets=100': 0.56469292,
            'map/iou=0.75/area=all/maxDets=100': 0.30133106,
            'map/iou=0.50:0.95/area=small/maxDets=100': 0.38737403,
            'map/iou=0.50:0.95/area=medium/maxDets=100': 0.31018272,
            'map/iou=0.50:0.95/area=large/maxDets=100': 0.32693391,
            'mar/iou=0.50:0.95/area=all/maxDets=1': 0.27037258,
            'mar/iou=0.50:0.95/area=all/maxDets=10': 0.41759154,
            'mar/iou=0.50:0.95/area=all/maxDets=100': 0.41898236,
            'mar/iou=0.50:0.95/area=small/maxDets=100': 0.46944986,
            'mar/iou=0.50:0.95/area=medium/maxDets=100': 0.37675923,
            'mar/iou=0.50:0.95/area=large/maxDets=100': 0.38147151
        }

        for key, item in expected.items():
            non_mean_key = key[1:]
            self.assertIsInstance(result[non_mean_key], np.ndarray)
            self.assertEqual(result[non_mean_key].shape, (76,))
            np.testing.assert_almost_equal(
                result[key], expected[key], decimal=5)


testing.run_module(__name__, __file__)
