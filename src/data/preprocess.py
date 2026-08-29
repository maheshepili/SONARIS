"""Content-aware preprocessing shared by dataset preparation and inference."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def preprocess_sonar_image(image: Image.Image, low_percentile: float = 1.0, high_percentile: float = 99.0) -> Image.Image:
    """Return a contrast-normalized RGB sonar image without changing the source.

    Sonar rasters in this dataset use P6 image content despite mixed filename
    suffixes. The transform uses the decoded pixels, converts them to grayscale,
    clips extreme intensities, and linearly stretches the remaining range.
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.float32)
    low, high = np.percentile(pixels, (low_percentile, high_percentile))
    if high <= low:
        normalized = np.zeros_like(pixels, dtype=np.uint8)
    else:
        normalized = np.clip((pixels - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L").convert("RGB")
