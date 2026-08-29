import unittest
from pathlib import Path

from src.models.inference import result_payload


class InferencePayloadTests(unittest.TestCase):
    def test_payload_is_machine_readable_shape(self):
        detection = {"class": "Pipeline", "confidence": 0.9, "bbox": [1, 2, 3, 4], "anomaly_score": 0.1, "priority": "LOW", "reason": "Prototype heuristic."}
        payload = result_payload(Path("sample.pbm"), [detection])
        self.assertEqual(payload["image"], "sample.pbm")
        self.assertEqual(payload["detections"][0]["class"], "Pipeline")
        self.assertIn("reason", payload["detections"][0])
        self.assertEqual(payload["anomaly_score_type"], "HEURISTIC_ANOMALY_SCORE")
