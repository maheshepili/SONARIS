"""Extract connected anomalous regions from an autoencoder heatmap."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


DEFAULT_PATCH_THRESHOLD = 0.0038187976460903883
DEFAULT_MIN_REGION_AREA = 64


def _mask_sonar_artifacts(heatmap: np.ndarray) -> np.ndarray:
    """Suppress the same border and centerline artifacts as heatmap generation."""
    masked = heatmap.copy()
    _, width = masked.shape

    edge = max(1, int(round(width * 0.05)))
    center_width = max(1, int(round(width * 0.06)))
    center_start = max(0, (width - center_width) // 2)
    center_end = min(width, center_start + center_width)

    masked[:, :edge] = 0.0
    masked[:, width - edge:] = 0.0
    masked[:, center_start:center_end] = 0.0
    return masked


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected pixel coordinates for each component in ``mask``."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[neighbor_y, neighbor_x] and not visited[neighbor_y, neighbor_x]:
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))

        components.append(np.asarray(pixels, dtype=np.intp))

    return components


def _component_bbox(pixels: np.ndarray) -> list[int]:
    """Return the full xywh bounds of one connected component."""
    ys, xs = pixels[:, 0], pixels[:, 1]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _tightened_component_bbox(pixels: np.ndarray, heatmap: np.ndarray) -> list[int]:
    """Bound the strongest portion of a component, falling back when needed.

    The threshold combines the component's upper-quartile error with 75% of
    its maximum error. This keeps diffuse thresholded tails out of a candidate
    while preserving a full component whose errors are uniformly strong.
    """
    original_bbox = _component_bbox(pixels)
    ys, xs = pixels[:, 0], pixels[:, 1]
    errors = heatmap[ys, xs]
    cutoff = max(float(np.percentile(errors, 75)), float(errors.max()) * 0.75)
    strongest_mask = np.zeros_like(heatmap, dtype=bool)
    strongest_mask[ys[errors >= cutoff], xs[errors >= cutoff]] = True
    strongest_components = _connected_components(strongest_mask)
    if not strongest_components:
        return original_bbox
    strongest_component = max(
        strongest_components,
        key=lambda component: (
            float(heatmap[component[:, 0], component[:, 1]].max()),
            len(component),
        ),
    )
    return _component_bbox(strongest_component)


def regions_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = DEFAULT_PATCH_THRESHOLD,
    min_region_area: int = DEFAULT_MIN_REGION_AREA,
    tighten: bool = True,
) -> list[dict[str, float | int | list[int]]]:
    """Return meaningful thresholded regions with their spatial error statistics.

    Bounding boxes use ``[x, y, width, height]`` coordinates. Area is the
    number of anomalous pixels in the connected component, not bbox area.
    """
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a two-dimensional array")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if min_region_area < 1:
        raise ValueError("min_region_area must be at least one pixel")

    masked_heatmap = _mask_sonar_artifacts(heatmap)
    mask = masked_heatmap > threshold
    regions: list[dict[str, float | int | list[int]]] = []

    for pixels in _connected_components(mask):
        area = len(pixels)
        if area < min_region_area:
            continue

        ys, xs = pixels[:, 0], pixels[:, 1]
        errors = masked_heatmap[ys, xs]
        regions.append(
            {
                "bbox": (
                    _tightened_component_bbox(pixels, masked_heatmap)
                    if tighten else _component_bbox(pixels)
                ),
                "area": area,
                "max_error": round(float(errors.max()), 8),
                "mean_error": round(float(errors.mean()), 8),
            }
        )

    return sorted(regions, key=lambda region: float(region["max_error"]), reverse=True)


def extract_candidate_regions(
    image_path: Path,
    checkpoint: Path,
    threshold: float = DEFAULT_PATCH_THRESHOLD,
    min_region_area: int = DEFAULT_MIN_REGION_AREA,
    tighten: bool = True,
) -> list[dict[str, float | int | list[int]]]:
    """Generate the masked autoencoder heatmap and extract candidate regions."""
    from src.anomaly.heatmap import anomaly_map

    _, heatmap = anomaly_map(image_path, checkpoint, threshold)
    return regions_from_heatmap(heatmap, threshold, min_region_area, tighten)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("reports/anomaly/sonar_autoencoder.pt"),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_PATCH_THRESHOLD)
    parser.add_argument("--min-region-area", type=int, default=DEFAULT_MIN_REGION_AREA)
    args = parser.parse_args()

    if not args.image.exists() or not args.checkpoint.exists():
        raise SystemExit("Image or anomaly checkpoint does not exist.")

    regions = extract_candidate_regions(
        args.image,
        args.checkpoint,
        args.threshold,
        args.min_region_area,
    )
    print(
        json.dumps(
            {
                "image": str(args.image),
                "threshold": args.threshold,
                "min_region_area": args.min_region_area,
                "regions": regions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
