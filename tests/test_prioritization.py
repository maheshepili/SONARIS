import unittest

from src.inference.prioritization import (
    HIGH_PRIORITY_ANOMALY_EXCESS_RATIO,
    LOW_PRIORITY_PIPELINE_CONFIDENCE,
    prioritize_evidence,
)


class PrioritizationTests(unittest.TestCase):
    def test_no_detection_or_anomaly_is_low(self):
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=False,
            anomaly_excess_over_threshold=0.0,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.finding, "No Significant Finding")
        self.assertEqual(result.priority, "LOW")
        self.assertFalse(result.requires_expert_verification)

    def test_confident_pipeline_only_is_low(self):
        result = prioritize_evidence(
            pipeline_detected=True,
            yolo_confidence=LOW_PRIORITY_PIPELINE_CONFIDENCE,
            anomaly_detected=False,
            anomaly_excess_over_threshold=0.0,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.finding, "Known Pipeline")
        self.assertEqual(result.priority, "LOW")
        self.assertFalse(result.requires_expert_verification)

    def test_lower_confidence_pipeline_only_is_medium(self):
        result = prioritize_evidence(
            pipeline_detected=True,
            yolo_confidence=LOW_PRIORITY_PIPELINE_CONFIDENCE - 0.01,
            anomaly_detected=False,
            anomaly_excess_over_threshold=0.0,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.priority, "MEDIUM")
        self.assertFalse(result.requires_expert_verification)

    def test_pipeline_without_confidence_is_medium(self):
        result = prioritize_evidence(
            pipeline_detected=True,
            yolo_confidence=None,
            anomaly_detected=False,
            anomaly_excess_over_threshold=None,
            image_anomaly_threshold=None,
        )

        self.assertEqual(result.priority, "MEDIUM")

    def test_strong_anomaly_only_is_high_and_requires_expert(self):
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=True,
            anomaly_excess_over_threshold=0.05,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.finding, "Potential Anomaly")
        self.assertEqual(result.priority, "HIGH")
        self.assertTrue(result.requires_expert_verification)

    def test_non_strong_anomaly_only_is_medium_and_requires_expert(self):
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=True,
            anomaly_excess_over_threshold=0.01,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.priority, "MEDIUM")
        self.assertTrue(result.requires_expert_verification)

    def test_anomaly_without_numeric_excess_is_medium(self):
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=True,
            anomaly_excess_over_threshold=None,
            image_anomaly_threshold=None,
        )

        self.assertEqual(result.finding, "Potential Anomaly")
        self.assertEqual(result.priority, "MEDIUM")
        self.assertTrue(result.requires_expert_verification)

    def test_pipeline_and_anomaly_is_high_regardless_of_excess(self):
        result = prioritize_evidence(
            pipeline_detected=True,
            yolo_confidence=0.99,
            anomaly_detected=True,
            anomaly_excess_over_threshold=0.0,
            image_anomaly_threshold=0.1,
        )

        self.assertEqual(result.finding, "Known Pipeline + Potential Anomaly")
        self.assertEqual(result.priority, "HIGH")
        self.assertTrue(result.requires_expert_verification)

    def test_candidate_regions_are_preserved_as_evidence(self):
        regions = [{"bbox": [1, 2, 3, 4]}, {"bbox": [5, 6, 7, 8]}]
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=True,
            anomaly_excess_over_threshold=0.01,
            image_anomaly_threshold=0.1,
            candidate_regions=regions,
        )

        self.assertEqual(result.evidence["candidate_region_count"], 2)
        self.assertEqual(result.evidence["candidate_regions"], regions)
        self.assertEqual(result.priority, "MEDIUM")

    def test_invalid_numeric_evidence_is_rejected(self):
        common = {
            "pipeline_detected": False,
            "anomaly_detected": False,
            "candidate_regions": None,
        }
        with self.assertRaises(ValueError):
            prioritize_evidence(
                **common,
                yolo_confidence=1.01,
                anomaly_excess_over_threshold=None,
                image_anomaly_threshold=None,
            )
        with self.assertRaises(ValueError):
            prioritize_evidence(
                **common,
                yolo_confidence=None,
                anomaly_excess_over_threshold=-0.01,
                image_anomaly_threshold=0.1,
            )
        with self.assertRaises(ValueError):
            prioritize_evidence(
                **common,
                yolo_confidence=None,
                anomaly_excess_over_threshold=0.0,
                image_anomaly_threshold=0.0,
            )

    def test_result_serializes_to_required_fields(self):
        result = prioritize_evidence(
            pipeline_detected=False,
            yolo_confidence=None,
            anomaly_detected=False,
            anomaly_excess_over_threshold=None,
            image_anomaly_threshold=None,
        )

        self.assertEqual(
            set(result.to_dict()),
            {"finding", "priority", "requires_expert_verification", "reason", "evidence"},
        )


if __name__ == "__main__":
    unittest.main()
