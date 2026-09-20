import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src.anomaly.evaluate_candidate_localization import (
    box_iou,
    evaluate_candidate_localization,
)


class CandidateLocalizationTests(unittest.TestCase):
    def test_iou_for_known_overlapping_boxes(self):
        self.assertAlmostEqual(
            box_iou((0, 0, 10, 10), (5, 5, 15, 15)),
            25 / 175,
        )

    def test_iou_is_zero_for_non_overlapping_boxes(self):
        self.assertEqual(box_iou((0, 0, 2, 2), (3, 3, 5, 5)), 0.0)

    def test_evaluator_passes_tighten_setting_to_region_extraction(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (100, 100)).save(images_dir / "sample.png")
            (labels_dir / "sample.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            for tighten in (True, False):
                with self.subTest(tighten=tighten), patch(
                    "src.anomaly.evaluate_candidate_localization.extract_candidate_regions",
                    return_value=[],
                ) as extract:
                    summary = evaluate_candidate_localization(
                    images_dir, labels_dir, root / "checkpoint.pt", tighten=tighten,
                    )

                extract.assert_called_once_with(
                    images_dir / "sample.png",
                    root / "checkpoint.pt",
                    threshold=0.0038187976460903883,
                    tighten=tighten,
                    tightening_factor=0.60,
                )
                self.assertEqual(summary["tighten"], tighten)

    def test_evaluator_passes_tightening_factor_to_region_extraction(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (100, 100)).save(images_dir / "sample.png")
            (labels_dir / "sample.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            with patch(
                "src.anomaly.evaluate_candidate_localization.extract_candidate_regions",
                return_value=[],
            ) as extract:
                summary = evaluate_candidate_localization(
                    images_dir,
                    labels_dir,
                    root / "checkpoint.pt",
                    tightening_factor=0.6,
                )

            extract.assert_called_once_with(
                images_dir / "sample.png",
                root / "checkpoint.pt",
                threshold=0.0038187976460903883,
                tighten=True,
                tightening_factor=0.6,
            )
            self.assertEqual(summary["tightening_factor"], 0.6)


if __name__ == "__main__":
    unittest.main()
