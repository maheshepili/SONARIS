"""Summarize Drishti validation labels by confirmed sonar object class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASS_NAMES = {
    1: "Pipeline",
    2: "Shipwreck",
    3: "Ghost Net",
    4: "Mine Cylinder",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGES_DIR = Path("data/external/drishti/val/images")
DEFAULT_LABELS_DIR = Path("data/external/drishti/val/labels")


def summarize_labels(images_dir: Path, labels_dir: Path) -> dict[str, int | dict[str, dict[str, int]]]:
    """Count Drishti validation annotations and labelled images by class."""
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    image_stems = {path.stem for path in image_paths}
    summary = {
        name: {
            "class_id": class_id,
            "annotation_count": 0,
            "image_count": 0,
        }
        for class_id, name in CLASS_NAMES.items()
    }

    for label_path in sorted(labels_dir.glob("*.txt")):
        if label_path.stem not in image_stems:
            raise ValueError(f"Label has no matching image: {label_path}")

        classes_in_image: set[int] = set()
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            values = line.split()
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}"
                )

            try:
                class_id = int(values[0])
            except ValueError as error:
                raise ValueError(
                    f"Invalid class ID at {label_path}:{line_number}"
                ) from error

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class ID at {label_path}:{line_number}: {class_id}"
                )

            summary[CLASS_NAMES[class_id]]["annotation_count"] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            summary[CLASS_NAMES[class_id]]["image_count"] += 1

    return {
        "image_count": len(image_paths),
        "annotation_count": sum(
            values["annotation_count"] for values in summary.values()
        ),
        "classes": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    args = parser.parse_args()

    if not args.images_dir.is_dir() or not args.labels_dir.is_dir():
        raise SystemExit("Drishti validation image or label directory does not exist.")

    print(json.dumps(summarize_labels(args.images_dir, args.labels_dir), indent=2))


if __name__ == "__main__":
    main()
