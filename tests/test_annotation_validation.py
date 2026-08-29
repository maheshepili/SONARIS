import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_dataset import _validate_coco


class AnnotationValidationTests(unittest.TestCase):
    def test_counts_missing_category_and_out_of_bounds_boxes(self):
        payload = {
            "images": [{"id": 1, "width": 10, "height": 10}],
            "categories": [{"id": 1, "name": "Pipeline"}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
                {"image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]},
                {"image_id": 1, "category_id": 1, "bbox": [8, 0, 5, 5]},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "annotations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_validate_coco(path), 2)
