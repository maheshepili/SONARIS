import unittest

from src.anomaly.evaluate_candidate_localization import box_iou


class CandidateLocalizationTests(unittest.TestCase):
    def test_iou_for_known_overlapping_boxes(self):
        self.assertAlmostEqual(
            box_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25 / 175,
        )

    def test_iou_is_zero_for_non_overlapping_boxes(self):
        self.assertEqual(box_iou((0, 0, 2, 2), (3, 3, 5, 5)), 0.0)


if __name__ == "__main__":
    unittest.main()
