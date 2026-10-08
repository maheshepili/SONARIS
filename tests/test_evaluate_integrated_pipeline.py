import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from src.inference.evaluate_integrated_pipeline import (
    discover_images,
    is_smaller_than_patch_size,
    main,
)


class EvaluateIntegratedPipelineTests(unittest.TestCase):
    def test_discover_images_includes_supported_extensions_in_sorted_order(self):
        with TemporaryDirectory() as temporary_directory:
            images_directory = Path(temporary_directory)
            for filename in (
                "zeta.png",
                "alpha.jpg",
                "middle.jpeg",
                "ignored.bmp",
            ):
                (images_directory / filename).touch()

            images = discover_images(images_directory)

        self.assertEqual(
            [image.name for image in images],
            ["alpha.jpg", "middle.jpeg", "zeta.png"],
        )

    def test_identifies_images_smaller_than_autoencoder_patch_size(self):
        with TemporaryDirectory() as temporary_directory:
            images_directory = Path(temporary_directory)
            undersized_width = images_directory / "undersized_width.png"
            undersized_height = images_directory / "undersized_height.png"
            minimum_size = images_directory / "minimum_size.png"
            Image.new("L", (127, 128)).save(undersized_width)
            Image.new("L", (128, 127)).save(undersized_height)
            Image.new("L", (128, 128)).save(minimum_size)

            self.assertTrue(
                is_smaller_than_patch_size(undersized_width, 128)
            )
            self.assertTrue(
                is_smaller_than_patch_size(undersized_height, 128)
            )
            self.assertFalse(
                is_smaller_than_patch_size(minimum_size, 128)
            )

    def test_skips_undersized_images_and_reports_them_in_summary(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images_directory = root / "images"
            images_directory.mkdir()
            valid_image = images_directory / "valid.png"
            skipped_image = images_directory / "too_small.jpg"
            Image.new("L", (128, 128)).save(valid_image)
            Image.new("L", (127, 128)).save(skipped_image)
            yolo_weights = root / "weights.pt"
            checkpoint = root / "checkpoint.pt"
            output = root / "summary.json"
            yolo_weights.touch()
            checkpoint.touch()

            with (
                patch(
                    "src.inference.evaluate_integrated_pipeline.load_model",
                    return_value=(object(), 128),
                ),
                patch(
                    "src.inference.evaluate_integrated_pipeline.run_inference",
                    return_value=({"detections": []}, None),
                ) as run_inference,
                patch(
                    "src.inference.evaluate_integrated_pipeline.score_image",
                    return_value={
                        "anomalous": False,
                        "max_reconstruction_error": 0.01,
                        "mean_reconstruction_error": 0.005,
                    },
                ) as score_image,
                patch.object(
                    sys,
                    "argv",
                    [
                        "evaluate_integrated_pipeline.py",
                        "--images",
                        str(images_directory),
                        "--yolo-weights",
                        str(yolo_weights),
                        "--anomaly-checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                    ],
                ),
            ):
                main()

            summary = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["images_evaluated"], 1)
        self.assertEqual(summary["skipped_image_count"], 1)
        self.assertEqual(summary["skipped_images"], ["too_small.jpg"])
        run_inference.assert_called_once()
        score_image.assert_called_once()


if __name__ == "__main__":
    unittest.main()
