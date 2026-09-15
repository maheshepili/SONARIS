from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.anomaly.score_reconstruction import load_model, score_image


DEFAULT_THRESHOLD = 0.028139928355813026
DEFAULT_CHECKPOINT = Path("reports/anomaly/sonar_autoencoder.pt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find validation images flagged by the image-level anomaly threshold."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    args = parser.parse_args()

    if not args.images.exists():
        raise SystemExit(
            f"Image directory does not exist: {args.images}"
        )

    if not args.checkpoint.exists():
        raise SystemExit(
            f"Checkpoint does not exist: {args.checkpoint}"
        )

    images = sorted(args.images.glob("*.png"))

    if not images:
        raise SystemExit(f"No PNG images found in {args.images}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Images: {len(images)}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")
    print()

    model, patch_size = load_model(
        args.checkpoint,
        device,
    )

    flagged = []

    for index, image_path in enumerate(images, start=1):
        print(
            f"\rScoring {index}/{len(images)}",
            end="",
            flush=True,
        )

        result = score_image(
            image_path=image_path,
            checkpoint=args.checkpoint,
            threshold=args.threshold,
            model=model,
            patch_size=patch_size,
            device=device,
        )

        if bool(result["anomalous"]):
            flagged.append(result)

    print("\n")

    print(f"Flagged images: {len(flagged)}")
    print("=" * 70)

    for result in flagged:
        print(
            f"{Path(result['image']).name} | "
            f"max={result['max_reconstruction_error']:.8f} | "
            f"mean={result['mean_reconstruction_error']:.8f} | "
            f"excess={result['excess_over_threshold']:.8f}"
        )


if __name__ == "__main__":
    main()
