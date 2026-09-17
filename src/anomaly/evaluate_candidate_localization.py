"""Evaluate Ghost Net localization by reconstruction-error candidate regions.

This evaluator is read-only: it uses DRISHTI validation labels and prints JSON
without saving images, changing the model, or changing the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.anomaly.extract_candidate_regions import extract_candidate_regions


GHOST_NET_CLASS_ID = 3
DEFAULT_IMAGES_DIR = Path("data/external/drishti/val/images")
DEFAULT_LABELS_DIR = Path("data/external/drishti/val/labels")
DEFAULT_CHECKPOINT = Path("reports/anomaly/sonar_autoencoder.pt")
DEFAULT_PATCH_THRESHOLD = 0.0038187976460903883
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Box = tuple[float, float, float, float]


def xywh_to_xyxy(box: Box) -> Box:
    """Convert a pixel ``(x, y, width, height)`` box to ``(x1, y1, x2, y2)``."""
    x, y, width, height = box
    return x, y, x + width, y + height


def box_iou(first: Box, second: Box) -> float:
    """Return IoU for two ``(x1, y1, x2, y2)`` boxes."""
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def ground_truth_coverage(ground_truth: Box, candidate: Box) -> float:
    """Return the fraction of a GT box area covered by a candidate box."""
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth
    candidate_x1, candidate_y1, candidate_x2, candidate_y2 = candidate
    width = max(0.0, min(gt_x2, candidate_x2) - max(gt_x1, candidate_x1))
    height = max(0.0, min(gt_y2, candidate_y2) - max(gt_y1, candidate_y1))
    gt_area = max(0.0, gt_x2 - gt_x1) * max(0.0, gt_y2 - gt_y1)
    return width * height / gt_area if gt_area else 0.0


def load_ghost_net_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    """Load valid class-3 normalized YOLO boxes as pixel xyxy boxes."""
    boxes: list[Box] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            x_center, y_center, box_width, box_height = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from error
        if class_id != GHOST_NET_CLASS_ID:
            continue

        x = max(0.0, (x_center - box_width / 2) * width)
        y = max(0.0, (y_center - box_height / 2) * height)
        right = min(float(width), (x_center + box_width / 2) * width)
        bottom = min(float(height), (y_center + box_height / 2) * height)
        if right > x and bottom > y:
            boxes.append(xywh_to_xyxy((x, y, right - x, bottom - y)))
    return boxes


def evaluate_candidate_localization(
    images_dir: Path,
    labels_dir: Path,
    checkpoint: Path,
    limit: int | None = None,
) -> dict:
    """Evaluate class-3 images sequentially to keep memory use bounded."""
    per_image: list[dict] = []
    best_ious: list[float] = []
    best_coverages: list[float] = []
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Image has no matching label: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth = load_ghost_net_boxes(label_path, width, height)
        if not ground_truth:
            continue
        if limit is not None and len(per_image) >= limit:
            break

        regions = extract_candidate_regions(
            image_path,
            checkpoint,
            threshold=DEFAULT_PATCH_THRESHOLD,
        )
        candidates = [
            xywh_to_xyxy(tuple(float(value) for value in region["bbox"]))
            for region in regions
        ]
        box_results = []
        for gt_box in ground_truth:
            if candidates:
                best_candidate = max(candidates, key=lambda candidate: box_iou(gt_box, candidate))
                best_iou = box_iou(gt_box, best_candidate)
                coverage = ground_truth_coverage(gt_box, best_candidate)
            else:
                best_iou = 0.0
                coverage = 0.0
            best_ious.append(best_iou)
            best_coverages.append(coverage)
            box_results.append({
                "ground_truth_box_xyxy": list(gt_box),
                "best_iou": best_iou,
                "best_candidate_coverage": coverage,
            })

        per_image.append({
            "image": str(image_path),
            "ground_truth_boxes": len(ground_truth),
            "candidate_regions": len(candidates),
            "box_results": box_results,
        })

    return {
        "class_id": GHOST_NET_CLASS_ID,
        "patch_threshold": DEFAULT_PATCH_THRESHOLD,
        "images_evaluated": len(per_image),
        "ground_truth_boxes": len(best_ious),
        "images_with_at_least_one_candidate": sum(
            image["candidate_regions"] > 0 for image in per_image
        ),
        "gt_boxes_iou_at_least_0_1": sum(iou >= 0.1 for iou in best_ious),
        "gt_boxes_iou_at_least_0_25": sum(iou >= 0.25 for iou in best_ious),
        "gt_boxes_iou_at_least_0_5": sum(iou >= 0.5 for iou in best_ious),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "median_best_iou": float(np.median(best_ious)) if best_ious else 0.0,
        "mean_best_candidate_coverage": float(np.mean(best_coverages)) if best_coverages else 0.0,
        "images": per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, help="Maximum class-3 images to evaluate.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not args.images_dir.is_dir() or not args.labels_dir.is_dir():
        raise SystemExit("Drishti validation image or label directory does not exist.")
    if not args.checkpoint.is_file():
        raise SystemExit("SONARIS autoencoder checkpoint does not exist.")

    print(json.dumps(evaluate_candidate_localization(
        args.images_dir, args.labels_dir, args.checkpoint, args.limit,
    ), indent=2))


if __name__ == "__main__":
    main()
