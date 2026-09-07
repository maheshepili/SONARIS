"""Calibrate a reconstruction-error threshold from held-out sonar background patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.anomaly.autoencoder import SonarAutoencoder, reconstruction_error
from src.anomaly.train_autoencoder import SonarBackgroundPatches


def collect_errors(model: SonarAutoencoder, dataset: SonarBackgroundPatches, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        chunks.append(reconstruction_error(model, batch.to(device)).cpu().numpy())
    return np.concatenate(chunks)


def calibrate(args: argparse.Namespace) -> dict[str, float | int | str]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SonarAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_set = SonarBackgroundPatches(Path(args.train_images), Path(args.train_labels), args.patch_size, args.patches_per_image, args.seed)
    val_set = SonarBackgroundPatches(Path(args.val_images), Path(args.val_labels), args.patch_size, args.patches_per_image, args.seed + 100000)
    train_errors = collect_errors(model, train_set, args.batch_size, device)
    val_errors = collect_errors(model, val_set, args.batch_size, device)

    threshold = float(np.percentile(train_errors, args.percentile))
    result = {
        "device": str(device),
        "train_patches": int(len(train_errors)),
        "validation_patches": int(len(val_errors)),
        "percentile": float(args.percentile),
        "threshold": threshold,
        "train_mean_error": float(train_errors.mean()),
        "train_p95_error": float(np.percentile(train_errors, 95)),
        "validation_mean_error": float(val_errors.mean()),
        "validation_p95_error": float(np.percentile(val_errors, 95)),
        "validation_fraction_above_threshold": float(np.mean(val_errors > threshold)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--train-images", type=Path, default=Path("data/processed/sonaris_detection/images/train"))
    parser.add_argument("--train-labels", type=Path, default=Path("data/processed/sonaris_detection/labels/train"))
    parser.add_argument("--val-images", type=Path, default=Path("data/processed/sonaris_detection/images/val"))
    parser.add_argument("--val-labels", type=Path, default=Path("data/processed/sonaris_detection/labels/val"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/calibration.json"))
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
