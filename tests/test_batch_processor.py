import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.inference.batch_processor import DEFAULT_BATCH_RESULT_NAME, process_batch


class BatchProcessorTests(unittest.TestCase):
    def test_batch_output_preserves_integrated_prioritization_and_localization(self):
        candidate_regions = [{"bbox": [10, 20, 30, 40], "max_error": 0.04}]
        evidence = {
            "pipeline_detected": True,
            "anomaly_detected": True,
            "candidate_regions": candidate_regions,
        }
        integrated_result = {
            "finding": "Known Pipeline + Potential Anomaly",
            "priority": "HIGH",
            "reason": "Both evidence sources require expert verification.",
            "requires_expert_verification": True,
            "evidence": evidence,
            "known_object_detection": {
                "detected": True,
                "detections": [{"class": "Pipeline", "confidence": 0.9}],
            },
            "anomaly_analysis": {
                "detected": True,
                "candidate_regions": candidate_regions,
            },
        }

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "sample.png").touch()

            with patch(
                "src.inference.batch_processor.run_integrated_pipeline",
                return_value=integrated_result,
            ):
                result = process_batch(input_dir, output_dir)

            record = result["images"][0]
            saved_record = json.loads(
                (output_dir / DEFAULT_BATCH_RESULT_NAME).read_text(encoding="utf-8")
            )["images"][0]

        for batch_record in (record, saved_record):
            self.assertEqual(batch_record["priority"], "HIGH")
            self.assertEqual(
                batch_record["reason"],
                "Both evidence sources require expert verification.",
            )
            self.assertEqual(batch_record["evidence"], evidence)
            self.assertEqual(batch_record["candidate_regions"], candidate_regions)
            self.assertTrue(batch_record["requires_expert_verification"])


if __name__ == "__main__":
    unittest.main()
