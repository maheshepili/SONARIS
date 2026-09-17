"""Measure how well SONARIS anomaly heatmaps localize Drishti validation boxes.

The evaluator is read-only: it prints JSON and does not save heatmaps or alter
the dataset.  An anomalous pixel is one whose masked reconstruction-error map
is strictly greater than the calibrated patch threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.evaluate_drishti_anomalies import reconstruction_heatmap
from src.anomaly.score_reconstruction import load_model
from src.data.preprocess import preprocess_sonar_image


DEFAULT_IMAGES_DIR = Path("data/external/drishti/val/images")
DEFAULT_LABELS_DIR = Path("data/external/drishti/val/labels")
DEFAULT_CHECKPOINT = Path("reports/anomaly/sonar_autoencoder.pt")
DEFAULT_PATCH_THRESHOLD = 0.0038187976460903883
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Box = tuple[int, int, int, int]


def load_boxes(label_path: Path, width: int, height: int, class_id: int | None) -> list[Box]:
    """Read valid YOLO boxes, optionally retaining only one class."""
    boxes: list[Box] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        try:
            label_class = int(fields[0])
            x_center, y_center, box_width, box_height = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from error
        if class_id is not None and label_class != class_id:
            continue

        x1 = max(0, int(round((x_center - box_width / 2) * width)))
        y1 = max(0, int(round((y_center - box_height / 2) * height)))
        x2 = min(width, int(round((x_center + box_width / 2) * width)))
        y2 = min(height, int(round((y_center + box_height / 2) * height)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def localization_metrics(heatmap: np.ndarray, boxes: list[Box], threshold: float) -> dict:
    """Calculate pixel localization and error statistics for one heatmap."""
    inside_mask = np.zeros_like(heatmap, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        inside_mask[y1:y2, x1:x2] = True

    anomalous_mask = heatmap > threshold
    anomalous_pixels = int(anomalous_mask.sum())
    anomalous_inside = int((anomalous_mask & inside_mask).sum())
    box_coverages = []
    for x1, y1, x2, y2 in boxes:
        box_anomalous = int(anomalous_mask[y1:y2, x1:x2].sum())
        box_area = (x2 - x1) * (y2 - y1)
        box_coverages.append({
            "box": [x1, y1, x2, y2],
            "anomalous_pixels": box_anomalous,
            "area_pixels": box_area,
            "anomalous_coverage": box_anomalous / box_area,
        })

    inside_errors = heatmap[inside_mask]
    outside_errors = heatmap[~inside_mask]
    return {
        "total_anomalous_pixels": anomalous_pixels,
        "anomalous_pixels_inside_boxes": anomalous_inside,
        "anomalous_pixels_inside_boxes_fraction": (
            anomalous_inside / anomalous_pixels if anomalous_pixels else 0.0
        ),
        "box_coverages": box_coverages,
        "mean_anomaly_error_inside_boxes": float(inside_errors.mean()) if inside_errors.size else 0.0,
        "mean_anomaly_error_outside_boxes": float(outside_errors.mean()) if outside_errors.size else 0.0,
    }


def evaluate_localization(
    images_dir: Path,
    labels_dir: Path,
    checkpoint: Path,
    threshold: float = DEFAULT_PATCH_THRESHOLD,
    class_id: int | None = None,
    limit: int | None = None,
) -> dict:
    """Evaluate eligible validation images sequentially with one loaded model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_model(checkpoint, device)
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    results: list[dict] = []
    skipped_small: list[str] = []
    skipped_without_boxes: list[str] = []
    missing_labels: list[str] = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(image_path.name)
            continue
        with Image.open(image_path) as source:
            width, height = preprocess_sonar_image(source).size
        if width < patch_size or height < patch_size:
            skipped_small.append(image_path.name)
            continue
        boxes = load_boxes(label_path, width, height, class_id)
        if not boxes:
            skipped_without_boxes.append(image_path.name)
            continue
        if limit is not None and len(results) >= limit:
            break

        # This helper is the established, masked overlap-averaged SONARIS map.
        heatmap = reconstruction_heatmap(image_path, model, patch_size, device)
        metrics = localization_metrics(heatmap, boxes, threshold)
        results.append({
            "image": str(image_path),
            "label": str(label_path),
            "boxes": len(boxes),
            **metrics,
        })

    total_anomalous = sum(row["total_anomalous_pixels"] for row in results)
    total_anomalous_inside = sum(row["anomalous_pixels_inside_boxes"] for row in results)
    coverages = [
        coverage["anomalous_coverage"]
        for row in results for coverage in row["box_coverages"]
    ]
    return {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "patch_size": patch_size,
        "patch_threshold": threshold,
        "class_id": class_id,
        "images_evaluated": len(results),
        "summary": {
            "anomalous_pixels_inside_boxes": total_anomalous_inside,
            "total_anomalous_pixels": total_anomalous,
            "anomalous_pixels_inside_boxes_fraction": (
                total_anomalous_inside / total_anomalous if total_anomalous else 0.0
            ),
            "mean_box_anomalous_coverage": float(np.mean(coverages)) if coverages else 0.0,
            "mean_anomaly_error_inside_boxes": float(np.mean([
                row["mean_anomaly_error_inside_boxes"] for row in results
            ])) if results else 0.0,
            "mean_anomaly_error_outside_boxes": float(np.mean([
                row["mean_anomaly_error_outside_boxes"] for row in results
            ])) if results else 0.0,
        },
        "images": results,
        "skipped_small_images": skipped_small,
        "skipped_images_without_selected_boxes": skipped_without_boxes,
        "images_missing_labels": missing_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--class-id", type=int, help="Evaluate only this YOLO class ID.")
    parser.add_argument("--limit", type=int, help="Maximum eligible images to evaluate.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.images_dir.is_dir() or not args.labels_dir.is_dir():
        raise SystemExit("Drishti validation image or label directory does not exist.")
    if not args.checkpoint.is_file():
        raise SystemExit("SONARIS autoencoder checkpoint does not exist.")

    print(json.dumps(evaluate_localization(
        args.images_dir, args.labels_dir, args.checkpoint,
        class_id=args.class_id, limit=args.limit,
    ), indent=2))


if __name__ == "__main__":
    main()
