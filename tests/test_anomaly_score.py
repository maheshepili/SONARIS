import unittest

from src.anomaly.anomaly_score import score_detection


class AnomalyScoreTests(unittest.TestCase):
    def test_low_confidence_is_more_suspicious(self):
        low, _, _ = score_detection(0.30, [100, 100, 100, 100], (1000, 500))
        high, _, _ = score_detection(0.90, [100, 100, 100, 100], (1000, 500))
        self.assertGreater(low, high)

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            score_detection(1.1, [0, 0, 1, 1], (10, 10))

    def test_malformed_bbox_is_rejected(self):
        with self.assertRaises(ValueError):
            score_detection(0.5, [0, 0, 1], (10, 10))
