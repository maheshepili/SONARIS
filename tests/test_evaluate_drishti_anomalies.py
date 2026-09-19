import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from src.anomaly.evaluate_drishti_anomalies import (
    CLASS_NAMES,
    batched_reconstruction_heatmap,
    load_class_boxes,
)


class OffsetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(patches))
        return patches + 0.25


class EvaluateDrishtiAnomaliesTests(unittest.TestCase):
    def _label_path(self, contents: str) -> tuple[TemporaryDirectory, Path]:
        directory = TemporaryDirectory()
        path = Path(directory.name) / "sample.txt"
        path.write_text(contents, encoding="utf-8")
        return directory, path

    def test_uses_unified_class_ids_zero_through_three(self):
        self.assertEqual(CLASS_NAMES, {
            0: "Pipeline",
            1: "Shipwreck",
            2: "Ghost Net",
            3: "Mine Cylinder",
        })

    def test_valid_bbox_is_accepted(self):
        directory, label_path = self._label_path("2 0.5 0.5 0.4 0.2\n")
        self.addCleanup(directory.cleanup)
        self.assertEqual(load_class_boxes(label_path, 100, 50)[2], [(30, 20, 70, 30)])

    def test_out_of_bounds_bbox_is_rejected_without_clipping(self):
        directory, label_path = self._label_path("0 0.95 0.5 0.2 0.2\n")
        self.addCleanup(directory.cleanup)
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            load_class_boxes(label_path, 100, 50)

    def test_missing_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing YOLO label"):
            load_class_boxes(Path("does-not-exist.txt"), 100, 50)

    def test_batched_heatmap_accumulates_overlapping_synthetic_patches(self):
        array = np.zeros((6, 6), dtype=np.float32)
        model = OffsetModel()
        heatmap = batched_reconstruction_heatmap(
            array,
            model,
            patch_size=4,
            device=torch.device("cpu"),
            batch_size=2,
        )

        # Every valid patch has MSE 0.25**2; only artifact-masked columns differ.
        expected = np.full((6, 6), 0.0625, dtype=np.float32)
        expected[:, 0] = 0.0
        expected[:, 2] = 0.0
        expected[:, 5] = 0.0
        np.testing.assert_allclose(heatmap, expected)
        self.assertEqual(model.batch_sizes, [2, 2])

    def test_batched_heatmap_rejects_non_positive_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            batched_reconstruction_heatmap(
                np.zeros((4, 4), dtype=np.float32),
                OffsetModel(),
                patch_size=4,
                device=torch.device("cpu"),
                batch_size=0,
            )
