import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from src.data.prepare_multiclass_dataset import (
    _drishti_records,
    external_validation_allowed,
    remap_drishti_labels,
    unique_filename,
)


class PrepareMulticlassDatasetTests(unittest.TestCase):
    def test_remaps_drishti_source_ids_and_preserves_coordinates(self):
        labels, ids = remap_drishti_labels("1 0.5 0.5 0.2 0.2\n4 0.4 0.4 0.1 0.1\n")
        self.assertEqual(ids, [0, 3])
        self.assertEqual(labels, "0 0.5 0.5 0.2 0.2\n3 0.4 0.4 0.1 0.1\n")

    def test_source_band_makes_colliding_sonaris_names_unique(self):
        original = "123.000.pbm"
        self.assertNotEqual(unique_filename("sonaris", "SSS_HF_images", original), unique_filename("sonaris", "SSS_LF_images", original))

    def test_leakage_risk_is_excluded_from_external_validation(self):
        self.assertFalse(external_validation_allowed("LEAKAGE_RISK"))
        self.assertTrue(external_validation_allowed("SAFE_EXTERNAL"))

    def _drishti_records_for(self, label: str):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "drishti"
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir()
            image_path = root / "images" / "sample.png"
            Image.new("L", (100, 50)).save(image_path)
            (root / "labels" / "sample.txt").write_text(label, encoding="utf-8")
            return _drishti_records(root, {}, include_for_training=False)

    def test_valid_drishti_bbox_is_accepted_for_external_validation(self):
        included, excluded = self._drishti_records_for("1 0.5 0.5 0.2 0.4\n")
        self.assertEqual(len(included), 1)
        self.assertEqual(included[0][0]["split"], "external_validation")
        self.assertEqual(excluded, [])

    def test_negative_or_outside_drishti_bbox_is_rejected(self):
        included, excluded = self._drishti_records_for("1 -0.1 0.5 0.2 0.2\n")
        self.assertEqual(included, [])
        self.assertEqual(excluded[0]["reason"], "OUT_OF_BOUNDS_BBOX")

    def test_bbox_extending_past_image_boundary_is_rejected_and_recorded(self):
        included, excluded = self._drishti_records_for("2 0.95 0.5 0.2 0.2\n")
        self.assertEqual(included, [])
        self.assertEqual(excluded, [{
            "original_filename": "sample.png",
            "source_dataset": "DRISHTI",
            "split": "external_validation",
            "unified_class_id": 1,
            "original_bbox": [0.95, 0.5, 0.2, 0.2],
            "image_width": 100,
            "image_height": 50,
            "reason": "OUT_OF_BOUNDS_BBOX",
        }])
