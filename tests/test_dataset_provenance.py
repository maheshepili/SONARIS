import tempfile
import unittest
from pathlib import Path

from src.data.check_dataset_provenance import check_provenance


class DatasetProvenanceTests(unittest.TestCase):
    def test_timestamp_crop_in_training_is_reported_as_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sonaris = root / "sonaris"
            raw = sonaris / "DATA" / "SSS_HF_images" / "Image"
            raw.mkdir(parents=True)
            (raw / "123.000.png").write_bytes(b"source")
            drishti = root / "drishti" / "images"
            drishti.mkdir(parents=True)
            (drishti / "bg_123.000_x500.png").write_bytes(b"crop")
            prepared = root / "prepared" / "images" / "train"
            prepared.mkdir(parents=True)
            (prepared / "SSS_HF_images_123.000.png").write_bytes(b"prepared")

            report = check_provenance(sonaris, root / "drishti", root / "prepared")

            relationship = report["relationships"][0]
            self.assertEqual(relationship["status"], "LEAKAGE_RISK")
            self.assertEqual(relationship["provenance"]["crop_x_offset"], 500)
            self.assertEqual(relationship["sonaris_candidates"][0]["current_split"], "train")

