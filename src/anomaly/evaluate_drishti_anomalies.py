"""Evaluate SONARIS reconstruction errors against Drishti validation labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

CLASS_NAMES = {
    0: "Pipeline",
    1: "Shipwreck",
    2: "Ghost Net",
    3: "Mine Cylinder",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGES_DIR = Path("data/processed/sonaris_multiclass_clean_v2/images/external_validation")
DEFAULT_LABELS_DIR = Path("data/processed/sonaris_multiclass_clean_v2/labels/external_validation")
DEFAULT_CHECKPOINT = Path("reports/anomaly/sonar_autoencoder.pt")
DEFAULT_IMAGE_THRESHOLD = 0.028139928355813026


def load_class_boxes(
    label_path: Path,
    width: int,
    height: int,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """Load strict, in-bounds unified-class YOLO boxes grouped by class ID."""
    if not label_path.is_file():
        raise ValueError(f"Missing YOLO label: {label_path}")
    boxes = {class_id: [] for class_id in CLASS_NAMES}

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")

        try:
            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(float, values[1:])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from error

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class ID at {label_path}:{line_number}: {class_id}"
            )

        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        if (
            not all(math.isfinite(value) for value in (x1, y1, x2, y2))
            or x1 < 0
            or y1 < 0
            or x2 > width
            or y2 > height
            or x2 <= x1
            or y2 <= y1
        ):
            raise ValueError(
                f"Out-of-bounds or degenerate YOLO bbox at {label_path}:{line_number}: "
                f"{values[1:]} -> ({x1}, {y1}, {x2}, {y2}) for image {width}x{height}"
            )

        pixel_box = tuple(int(round(value)) for value in (x1, y1, x2, y2))
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise ValueError(
                f"Degenerate YOLO bbox after pixel conversion at {label_path}:{line_number}: "
                f"{values[1:]} -> {pixel_box}"
            )
        boxes[class_id].append(pixel_box)

    return boxes


def reconstruction_heatmap(
    image_path: Path,
    model: SonarAutoencoder,
    patch_size: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return the existing masked, overlap-averaged reconstruction error map."""
    import numpy as np
    from src.data.preprocess import preprocess_sonar_image

    with Image.open(image_path) as source:
        image = preprocess_sonar_image(source).convert("L")

    array = np.asarray(image, dtype=np.float32) / 255.0
    return batched_reconstruction_heatmap(array, model, patch_size, device, batch_size)


def batched_reconstruction_heatmap(
    array: np.ndarray,
    model: SonarAutoencoder,
    patch_size: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return the masked overlap-averaged reconstruction error map for an image array.

    Patches retain the evaluator's existing row-major traversal order and are
    accumulated in that same order after reconstruction-error inference.
    """
    import numpy as np
    import torch

    from src.anomaly.autoencoder import reconstruction_error
    from src.anomaly.heatmap import mask_sonar_artifacts

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    height, width = array.shape
    stride = patch_size // 2
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    def accumulate_batch() -> None:
        if not patches:
            return
        tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
        errors = reconstruction_error(model, tensor).tolist()
        for (y, x), error in zip(positions, errors, strict=True):
            heatmap[y:y + patch_size, x:x + patch_size] += error
            counts[y:y + patch_size, x:x + patch_size] += 1.0
        patches.clear()
        positions.clear()

    with torch.no_grad():
        for y in range(0, max(1, height - patch_size + 1), stride):
            for x in range(0, max(1, width - patch_size + 1), stride):
                patch = array[y:y + patch_size, x:x + patch_size]
                if patch.shape != (patch_size, patch_size):
                    continue
                patches.append(patch)
                positions.append((y, x))
                if len(patches) == batch_size:
                    accumulate_batch()
        accumulate_batch()

    valid = counts > 0
    heatmap[valid] /= counts[valid]
    return mask_sonar_artifacts(heatmap)


def evaluate_drishti_anomalies(
    images_dir: Path,
    labels_dir: Path,
    checkpoint: Path,
    image_threshold: float = DEFAULT_IMAGE_THRESHOLD,
    batch_size: int = 64,
) -> dict:
    """Compare reconstruction errors inside and outside each Drishti class."""
    import numpy as np
    import torch

    from src.anomaly.score_reconstruction import load_model, score_image
    from src.data.preprocess import preprocess_sonar_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch_size = load_model(checkpoint, device)
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    measurements = {
        class_id: {"inside": [], "outside": [], "images": 0, "flagged": 0}
        for class_id in CLASS_NAMES
    }
    skipped_images: list[str] = []

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for image_number, image_path in enumerate(image_paths, start=1):
        print(f"[{image_number}/{len(image_paths)}] {image_path.name}")
        with Image.open(image_path) as source:
            width, height = preprocess_sonar_image(source).size
        if width < patch_size or height < patch_size:
            skipped_images.append(image_path.name)
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise ValueError(f"Image has no matching label: {image_path}")
        boxes_by_class = load_class_boxes(label_path, width, height)
        present_classes = [class_id for class_id, boxes in boxes_by_class.items() if boxes]
        if not present_classes:
            continue

        score = score_image(
            image_path,
            checkpoint,
            image_threshold,
            model=model,
            patch_size=patch_size,
            device=device,
        )
        heatmap = reconstruction_heatmap(
            image_path,
            model,
            patch_size,
            device,
            batch_size,
        )

        for class_id in present_classes:
            inside_mask = np.zeros_like(heatmap, dtype=bool)
            for x1, y1, x2, y2 in boxes_by_class[class_id]:
                inside_mask[y1:y2, x1:x2] = True

            class_measurements = measurements[class_id]
            class_measurements["inside"].extend(heatmap[inside_mask].tolist())
            class_measurements["outside"].extend(heatmap[~inside_mask].tolist())
            class_measurements["images"] += 1
            class_measurements["flagged"] += int(bool(score["anomalous"]))

    classes = {}
    for class_id, name in CLASS_NAMES.items():
        class_measurements = measurements[class_id]
        inside = np.asarray(class_measurements["inside"], dtype=np.float32)
        outside = np.asarray(class_measurements["outside"], dtype=np.float32)
        image_count = int(class_measurements["images"])
        classes[name] = {
            "class_id": class_id,
            "image_count": image_count,
            "inside_mean_error": round(float(inside.mean()), 8) if inside.size else 0.0,
            "inside_max_error": round(float(inside.max()), 8) if inside.size else 0.0,
            "outside_mean_error": round(float(outside.mean()), 8) if outside.size else 0.0,
            "outside_max_error": round(float(outside.max()), 8) if outside.size else 0.0,
            "image_level_anomaly_flag_rate": round(
                int(class_measurements["flagged"]) / image_count,
                6,
            ) if image_count else 0.0,
        }

    return {
        "images_evaluated": len(image_paths) - len(skipped_images),
        "checkpoint": str(checkpoint),
        "image_threshold": image_threshold,
        "classes": classes,
        "image_counts_by_unified_class": {
            str(class_id): classes[name]["image_count"]
            for class_id, name in CLASS_NAMES.items()
        },
        "skipped_image_count": len(skipped_images),
        "skipped_images": skipped_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-threshold", type=float, default=DEFAULT_IMAGE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.images_dir.is_dir() or not args.labels_dir.is_dir():
        raise SystemExit("Drishti validation image or label directory does not exist.")
    if not args.checkpoint.exists():
        raise SystemExit("SONARIS autoencoder checkpoint does not exist.")

    print(
        json.dumps(
            evaluate_drishti_anomalies(
                args.images_dir,
                args.labels_dir,
                args.checkpoint,
                args.image_threshold,
                args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
