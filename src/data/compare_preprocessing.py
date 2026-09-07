"""Compare SSS preprocessing variants on representative sonar images.

This experiment intentionally does not modify the production preprocessing pipeline.
It writes side-by-side candidate outputs so that preprocessing can be evaluated
before changing the training dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def percentile_normalize(image: Image.Image, low: float = 1.0, high: float = 99.0) -> Image.Image:
    gray = ImageOps.grayscale(image)
    pixels = np.asarray(gray, dtype=np.float32)
    lo, hi = np.percentile(pixels, (low, high))
    if hi <= lo:
        result = np.zeros_like(pixels, dtype=np.uint8)
    else:
        result = np.clip((pixels - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L").convert("RGB")


def clahe_like(image: Image.Image, tile_grid: int = 8, clip_limit: float = 2.0) -> Image.Image:
    """Apply CLAHE using OpenCV when available.

    OpenCV is optional so the repository's existing environment remains usable.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "CLAHE comparison requires OpenCV. Install it with: pip install opencv-python"
        ) from error

    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced, mode="L").convert("RGB")


def variants(image: Image.Image) -> dict[str, Image.Image]:
    baseline = percentile_normalize(image)
    clahe = clahe_like(baseline)
    # Median filtering is deliberately light; sonar edges and shadows must be preserved.
    denoised = clahe.filter(ImageFilter.MedianFilter(size=3))
    return {
        "original": image.convert("RGB"),
        "baseline": baseline,
        "clahe": clahe,
        "clahe_denoise": denoised,
    }


def compare(input_dir: Path, output_dir: Path) -> int:
    extensions = {".png", ".jpg", ".jpeg", ".pbm", ".ppm", ".bpm"}
    images = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    for image_path in images:
        with Image.open(image_path) as image:
            image_variants = variants(image)
        for name, rendered in image_variants.items():
            destination = output_dir / name / f"{image_path.stem}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(destination, "PNG")
        print(f"Compared: {image_path.name}")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/preprocessing_comparison"))
    args = parser.parse_args()
    count = compare(args.input_dir, args.output_dir)
    print(f"Processed {count} image(s). Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
