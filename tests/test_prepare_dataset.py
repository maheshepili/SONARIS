import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.prepare_dataset import prepare_dataset


class PrepareDatasetTests(unittest.TestCase):
    def test_converts_valid_labels_and_excludes_orphan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "raw"
            band = root / "SSS_LF_images"
            (band / "Image").mkdir(parents=True)
            labels = band / "YOLO_Annotation"
            labels.mkdir()
            (band / "COCO_Annotation").mkdir()
            Image.new("RGB", (20, 10)).save(band / "Image" / "sample.pbm", "PPM")
            (labels / "classes.txt").write_text("Pipeline\n")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (labels / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            hf = root / "SSS_HF_images"
            (hf / "Image").mkdir(parents=True)
            (hf / "YOLO_Annotation").mkdir()
            (hf / "COCO_Annotation").mkdir()
            (hf / "YOLO_Annotation" / "classes.txt").write_text("Pipeline\n")
            report = prepare_dataset(root, Path(temp) / "output", report_path=Path(temp) / "report.json")
            self.assertEqual(report.images_written, 1)
            self.assertEqual(report.orphan_label_files, 1)
            self.assertEqual(report.annotation_rows, 1)
