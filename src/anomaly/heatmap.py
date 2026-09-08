"""Generate spatial anomaly maps from autoencoder reconstruction error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.anomaly.autoencoder import reconstruction_error
from src.anomaly.score_reconstruction import load_model


def artifact_mask(
    height: int,
    width: int,
    border_fraction: float = 0.05,
    nadir_half_width_fraction: float = 0.03,
) -> np.ndarray:
    """Return True for regions excluded from anomaly consideration.

    Side-scan sonar images commonly contain strong acquisition artifacts near
    the image borders and around the nadir/centerline. These regions can have
    high reconstruction error even when they contain no object of interest.
    """
    mask = np.zeros((height, width), dtype=bool)
    border_y = int(round(height * border_fraction))
    border_x = int(round(width * border_fraction))
    if border_y > 0:
        mask[:border_y, :] = True
        mask[-border_y:, :] = True
    if border_x > 0:
        mask[:, :border_x] = True
        mask[:, -border_x:] = True

    center = width // 2
    half_width = max(1, int(round(width * nadir_half_width_fraction)))
    mask[:, max(0, center - half_width):min(width, center + half_width + 1)] = True
    return mask


def anomaly_map(image_path: Path, checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_model(checkpoint, device)
    with Image.open(image_path) as source:
        image = source.convert("L")
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


def save_heatmap(
    image_path: Path,
    checkpoint: Path,
    threshold: float,
    output: Path,
    border_fraction: float = 0.05,
    nadir_half_width_fraction: float = 0.03,
) -> dict[str, float | bool | str]:
    image, heatmap = anomaly_map(image_path, checkpoint)
    mask = artifact_mask(
        image.shape[0],
        image.shape[1],
        border_fraction=border_fraction,
        nadir_half_width_fraction=nadir_half_width_fraction,
    )
    filtered_heatmap = heatmap.copy()
    filtered_heatmap[mask] = 0.0

    normalized = np.clip(filtered_heatmap / max(threshold, 1e-8), 0.0, 2.0) / 2.0
    heat = (normalized * 255).astype(np.uint8)
    base = (image * 255).astype(np.uint8)
    # Grayscale composite: brighter values indicate stronger reconstruction error.
    overlay = np.maximum(base, heat)
    Image.fromarray(overlay).save(output)

    valid_heat = filtered_heatmap[~mask]
    result = {
        "image": str(image_path),
        "threshold": threshold,
        "border_fraction": border_fraction,
        "nadir_half_width_fraction": nadir_half_width_fraction,
        "max_error": round(float(valid_heat.max()) if valid_heat.size else 0.0, 8),
        "mean_error": round(float(valid_heat.mean()) if valid_heat.size else 0.0, 8),
        "anomalous": bool(float(valid_heat.max()) > threshold) if valid_heat.size else False,
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
    parser.add_argument("--border-fraction", type=float, default=0.05)
    parser.add_argument("--nadir-half-width-fraction", type=float, default=0.03)
    args = parser.parse_args()
    if not args.image.exists() or not args.checkpoint.exists():
        raise SystemExit("Image or anomaly checkpoint does not exist.")
    if not 0.0 <= args.border_fraction < 0.5:
        raise SystemExit("--border-fraction must be between 0 and 0.5.")
    if not 0.0 <= args.nadir_half_width_fraction < 0.5:
        raise SystemExit("--nadir-half-width-fraction must be between 0 and 0.5.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            save_heatmap(
                args.image,
                args.checkpoint,
                args.threshold,
                args.output,
                border_fraction=args.border_fraction,
                nadir_half_width_fraction=args.nadir_half_width_fraction,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
