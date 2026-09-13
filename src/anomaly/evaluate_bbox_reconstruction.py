"""Compare autoencoder reconstruction error inside and outside a YOLO box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.heatmap import anomaly_map


def load_pipeline_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        x1 = max(0, int(round((xc - bw / 2) * width)))
        y1 = max(0, int(round((yc - bh / 2) * height)))
        x2 = min(width, int(round((xc + bw / 2) * width)))
        y2 = min(height, int(round((yc + bh / 2) * height)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--threshold", type=float, default=0.0038187976460903883)
    args = parser.parse_args()

    if not args.image.exists() or not args.label.exists() or not args.checkpoint.exists():
        raise SystemExit("Image, label, or anomaly checkpoint does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(args.image) as source:
        width, height = source.size

    _, heatmap = anomaly_map(args.image, args.checkpoint, args.threshold)
    boxes = load_pipeline_boxes(args.label, width, height)
    if not boxes:
        raise SystemExit("No valid YOLO boxes found in the label file.")

    inside_values: list[float] = []
    inside_masks = np.zeros_like(heatmap, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        inside_masks[y1:y2, x1:x2] = True
        inside_values.extend(heatmap[y1:y2, x1:x2].ravel().tolist())

    outside_values = heatmap[~inside_masks]
    inside = np.asarray(inside_values, dtype=np.float32)
    outside = np.asarray(outside_values, dtype=np.float32)

    result = {
        "image": str(args.image),
        "label": str(args.label),
        "boxes": len(boxes),
        "threshold": args.threshold,
        "inside_mean_error": round(float(inside.mean()), 8),
        "inside_max_error": round(float(inside.max()), 8),
        "inside_fraction_above_threshold": round(float((inside > args.threshold).mean()), 6),
        "outside_mean_error": round(float(outside.mean()), 8),
        "outside_max_error": round(float(outside.max()), 8),
        "outside_fraction_above_threshold": round(float((outside > args.threshold).mean()), 6),
        "inside_to_outside_mean_ratio": round(float(inside.mean() / max(outside.mean(), 1e-8)), 4),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
