"""Calibrate a reconstruction-error threshold from held-out sonar background patches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.autoencoder import SonarAutoencoder, reconstruction_error
from src.anomaly.score_reconstruction import load_model
from src.anomaly.train_autoencoder import SonarBackgroundPatches
from src.data.preprocess import preprocess_sonar_image


def collect_errors(model: SonarAutoencoder, dataset: SonarBackgroundPatches, batch_size: int, device: torch.device) -> np.ndarray:
    """Collect reconstruction errors efficiently, preprocessing each image only once."""
    model.eval()
    all_errors: list[np.ndarray] = []
    patch_size = dataset.patch_size

    with torch.no_grad():
        for image_index, image_path in enumerate(dataset.images):
            with Image.open(image_path) as source:
                image = preprocess_sonar_image(source).convert("L")
            width, height = image.size
            if width < patch_size or height < patch_size:
                image = image.resize((max(width, patch_size), max(height, patch_size)), Image.Resampling.BILINEAR)
                width, height = image.size

            boxes = dataset._boxes(dataset.labels_dir / f"{image_path.stem}.txt", width, height)
            max_x, max_y = width - patch_size, height - patch_size
            patches: list[torch.Tensor] = []

            for patch_number in range(dataset.patches_per_image):
                index = image_index * dataset.patches_per_image + patch_number
                rng = random.Random(dataset.seed + index)
                chosen = None
                for _ in range(40):
                    x = rng.randint(0, max_x)
                    y = rng.randint(0, max_y)
                    candidate = (x, y, x + patch_size, y + patch_size)
                    if not any(dataset._overlap(candidate, box) for box in boxes):
                        chosen = candidate
                        break
                if chosen is None:
                    x = rng.randint(0, max_x)
                    y = rng.randint(0, max_y)
                    chosen = (x, y, x + patch_size, y + patch_size)

                patch = image.crop(chosen)
                array = np.asarray(patch, dtype=np.float32) / 255.0
                patches.append(torch.from_numpy(array).unsqueeze(0))

            for start in range(0, len(patches), batch_size):
                batch = torch.stack(patches[start:start + batch_size]).to(device)
                all_errors.append(reconstruction_error(model, batch).cpu().numpy())

    if not all_errors:
        raise ValueError("No calibration patches were generated.")
    return np.concatenate(all_errors)


def calibrate(args: argparse.Namespace) -> dict[str, float | int | str]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SonarAutoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_set = SonarBackgroundPatches(Path(args.train_images), Path(args.train_labels), args.patch_size, args.patches_per_image, args.seed)
    val_set = SonarBackgroundPatches(Path(args.val_images), Path(args.val_labels), args.patch_size, args.patches_per_image, args.seed + 100000)
    print(f"calibrating on {len(train_set.images)} train images + {len(val_set.images)} validation images using {device}")
    train_errors = collect_errors(model, train_set, args.batch_size, device)
    print("training errors collected")
    val_errors = collect_errors(model, val_set, args.batch_size, device)
    print("validation errors collected")

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
