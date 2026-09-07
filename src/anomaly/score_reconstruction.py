"""Score sonar patches with a calibrated autoencoder reconstruction threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.autoencoder import SonarAutoencoder, reconstruction_error


def load_model(checkpoint: Path, device: torch.device) -> tuple[SonarAutoencoder, int]:
    payload = torch.load(checkpoint, map_location=device)
    model = SonarAutoencoder().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, int(payload.get("patch_size", 128))


def score_image(image_path: Path, checkpoint: Path, threshold: float | None = None) -> dict[str, float | bool | str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_model(checkpoint, device)
    with Image.open(image_path) as source:
        image = source.convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width = array.shape
    scores: list[float] = []
    stride = patch_size // 2
    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            patch = array[y:y + patch_size, x:x + patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            scores.append(float(reconstruction_error(model, tensor)[0].item()))
    if not scores:
        raise ValueError(f"Image is smaller than the anomaly patch size: {image_path}")
    result: dict[str, float | bool | str] = {
        "image": str(image_path),
        "device": str(device),
        "patch_size": patch_size,
        "max_reconstruction_error": round(max(scores), 8),
        "mean_reconstruction_error": round(float(np.mean(scores)), 8),
    }
    if threshold is not None:
        result["threshold"] = threshold
        result["anomalous"] = bool(max(scores) > threshold)
        result["excess_over_threshold"] = round(max(scores) - threshold, 8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if not args.image.exists() or not args.checkpoint.exists():
        raise SystemExit("Image or anomaly checkpoint does not exist.")
    print(json.dumps(score_image(args.image, args.checkpoint, args.threshold), indent=2))


if __name__ == "__main__":
    main()
