"""Evaluate autoencoder anomaly scores across a validation image set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.heatmap import anomaly_map


def evaluate_image(image_path: Path, checkpoint: Path, threshold: float) -> dict[str, float | bool | str]:
    _, heatmap = anomaly_map(image_path, checkpoint, threshold)
    return {
        "image": str(image_path),
        "max_error": round(float(heatmap.max()), 8),
        "mean_error": round(float(heatmap.mean()), 8),
        "fraction_above_threshold": round(float((heatmap > threshold).mean()), 6),
        "anomalous": bool(float(heatmap.max()) > threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--threshold", type=float, default=0.0038187976460903883)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/validation_evaluation.json"))
    args = parser.parse_args()

    if not args.images.exists() or not args.checkpoint.exists():
        raise SystemExit("Image directory or anomaly checkpoint does not exist.")
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    image_paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".pbm"}
    )[: args.limit]
    if not image_paths:
        raise SystemExit("No supported images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"images={len(image_paths)}")

    results: list[dict[str, float | bool | str]] = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        results.append(evaluate_image(image_path, args.checkpoint, args.threshold))

    max_errors = np.asarray([float(r["max_error"]) for r in results], dtype=np.float32)
    mean_errors = np.asarray([float(r["mean_error"]) for r in results], dtype=np.float32)
    fractions = np.asarray([float(r["fraction_above_threshold"]) for r in results], dtype=np.float32)
    flagged = sum(bool(r["anomalous"]) for r in results)

    summary = {
        "threshold": args.threshold,
        "images_evaluated": len(results),
        "images_flagged": flagged,
        "flagged_fraction": round(flagged / len(results), 6),
        "max_error_mean": round(float(max_errors.mean()), 8),
        "max_error_p95": round(float(np.percentile(max_errors, 95)), 8),
        "mean_error_mean": round(float(mean_errors.mean()), 8),
        "fraction_above_threshold_mean": round(float(fractions.mean()), 6),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
