import unittest

from src.data.compare_sonar_annotations import coco_to_yolo, iou, _match_boxes


class SonarAnnotationComparisonTests(unittest.TestCase):
    def test_conversion_and_matching_distinguish_exact_missing_and_extra(self):
        converted = coco_to_yolo([20, 10, 40, 20], 100, 100)
        self.assertEqual(converted, (0.4, 0.2, 0.4, 0.2))
        result = _match_boxes([converted, (0.8, 0.8, 0.1, 0.1)], [converted, (0.1, 0.1, 0.1, 0.1)])
        self.assertEqual(result["exact"], 1)
        self.assertEqual(result["missing_coco_indices"], [1])
        self.assertEqual(result["extra_yolo_indices"], [1])
        self.assertEqual(iou(converted, converted), 1.0)

