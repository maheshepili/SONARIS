"""Calibrate an image-level reconstruction-error threshold."""

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


def image_max_error(
    model: SonarAutoencoder,
    image_path: Path,
    patch_size: int,
    device: torch.device,
) -> tuple[float, float, int]:

    with Image.open(image_path) as source:
        image = preprocess_sonar_image(source).convert("L")

    array = np.asarray(image, dtype=np.float32) / 255.0

    height, width = array.shape
    stride = patch_size // 2

    errors: list[float] = []

    with torch.no_grad():
        for y in range(
            0,
            max(1, height - patch_size + 1),
            stride,
        ):
            for x in range(
                0,
                max(1, width - patch_size + 1),
                stride,
            ):
                patch = array[
                    y:y + patch_size,
                    x:x + patch_size,
                ]

                if patch.shape != (patch_size, patch_size):
                    continue

                tensor = (
                    torch.from_numpy(patch)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                error = float(
                    reconstruction_error(model, tensor)[0].item()
                )

                errors.append(error)

    if not errors:
        raise ValueError(
            f"No valid patches found for image: {image_path}"
        )

    return (
        max(errors),
        float(np.mean(errors)),
        len(errors),
    )


def calibrate(args: argparse.Namespace) -> dict:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, patch_size = load_model(
        args.checkpoint,
        device,
    )

    image_paths = sorted(
        Path(args.images).glob("*.png")
    )

    if not image_paths:
        raise ValueError(
            f"No PNG images found in: {args.images}"
        )

    print(
        f"calibrating image-level threshold on "
        f"{len(image_paths)} images using {device}"
    )

    max_errors: list[float] = []
    mean_errors: list[float] = []
    patch_counts: list[int] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        max_error, mean_error, patch_count = image_max_error(
            model,
            image_path,
            patch_size,
            device,
        )

        max_errors.append(max_error)
        mean_errors.append(mean_error)
        patch_counts.append(patch_count)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} "
            f"max={max_error:.8f} "
            f"mean={mean_error:.8f}"
        )

    max_errors_array = np.asarray(
        max_errors,
        dtype=np.float32,
    )

    mean_errors_array = np.asarray(
        mean_errors,
        dtype=np.float32,
    )

    threshold = float(
        np.percentile(
            max_errors_array,
            args.percentile,
        )
    )

    result = {
        "device": str(device),
        "threshold_type": (
            "IMAGE_LEVEL_MAX_RECONSTRUCTION_ERROR"
        ),
        "percentile": float(args.percentile),
        "threshold": threshold,
        "images_evaluated": len(image_paths),
        "patch_size": patch_size,
        "patches_per_image_mean": float(
            np.mean(patch_counts)
        ),
        "image_max_mean": float(
            max_errors_array.mean()
        ),
        "image_max_p95": float(
            np.percentile(max_errors_array, 95)
        ),
        "image_max_p99": float(
            np.percentile(max_errors_array, 99)
        ),
        "image_max_max": float(
            max_errors_array.max()
        ),
        "image_mean_error_mean": float(
            mean_errors_array.mean()
        ),
    }

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nIMAGE-LEVEL CALIBRATION")
    print(json.dumps(result, indent=2))

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "reports/anomaly/sonar_autoencoder.pt"
        ),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/processed/"
            "sonaris_detection/images/val"
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/anomaly/"
            "image_level_calibration.json"
        ),
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint does not exist: "
            f"{args.checkpoint}"
        )

    if not args.images.exists():
        raise SystemExit(
            f"Image directory does not exist: "
            f"{args.images}"
        )

    calibrate(args)


if __name__ == "__main__":
    main()