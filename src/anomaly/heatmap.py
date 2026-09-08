"""Generate spatial anomaly maps from autoencoder reconstruction error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.autoencoder import SonarAutoencoder, reconstruction_error
from src.anomaly.score_reconstruction import load_model
from src.data.preprocess import preprocess_sonar_image


def anomaly_map(image_path: Path, checkpoint: Path, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_model(checkpoint, device)
    with Image.open(image_path) as source:
        # Match the same preprocessing used during anomaly training and detector inference.
        image = preprocess_sonar_image(source).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width = array.shape
    stride = patch_size // 2
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, max(1, height - patch_size + 1), stride):
            for x in range(0, max(1, width - patch_size + 1), stride):
                patch = array[y:y + patch_size, x:x + patch_size]
                if patch.shape != (patch_size, patch_size):
                    continue
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                error = float(reconstruction_error(model, tensor)[0].item())
                heatmap[y:y + patch_size, x:x + patch_size] += error
                counts[y:y + patch_size, x:x + patch_size] += 1.0
    valid = counts > 0
    heatmap[valid] /= counts[valid]
    return array, heatmap


def save_heatmap(image_path: Path, checkpoint: Path, threshold: float, output: Path) -> dict[str, float | bool | str]:
    image, heatmap = anomaly_map(image_path, checkpoint, threshold)
    normalized = np.clip(heatmap / max(threshold, 1e-8), 0.0, 2.0) / 2.0
    heat = (normalized * 255).astype(np.uint8)
    base = (image * 255).astype(np.uint8)
    overlay = np.maximum(base, heat)
    Image.fromarray(overlay).save(output)
    result = {
        "image": str(image_path),
        "threshold": threshold,
        "max_error": round(float(heatmap.max()), 8),
        "mean_error": round(float(heatmap.mean()), 8),
        "anomalous": bool(float(heatmap.max()) > threshold),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("reports/anomaly/sonar_autoencoder.pt"))
    parser.add_argument("--threshold", type=float, default=0.004991484340280294)
    parser.add_argument("--output", type=Path, default=Path("reports/anomaly/anomaly_heatmap.png"))
    args = parser.parse_args()
    if not args.image.exists() or not args.checkpoint.exists():
        raise SystemExit("Image or anomaly checkpoint does not exist.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(save_heatmap(args.image, args.checkpoint, args.threshold, args.output), indent=2))


if __name__ == "__main__":
    main()
