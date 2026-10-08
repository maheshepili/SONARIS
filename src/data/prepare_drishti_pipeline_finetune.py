"""Prepare a leakage-safe, Pipeline-only DRISHTI fine-tuning dataset.

The DRISHTI crops identify their source sonar frame in filenames such as
``bg_1693569219.799_x1000.jpg`` and ``pipe_1693569376.780_x1500.jpg``. This
module excludes every training crop whose source timestamp appears in the
existing external-validation set, then splits the remaining timestamp groups
(rather than individual crops).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


DRISHTI_TRAIN_DEFAULT = Path("data/external/drishti_train/train")
EXTERNAL_VALIDATION_DEFAULT = Path(
    "data/processed/sonaris_multiclass_clean_v2/images/external_validation"
)
OUTPUT_DEFAULT = Path("data/processed/drishti_pipeline_finetune")
SEED = 42
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pbm", ".pgm", ".png", ".ppm", ".tif", ".tiff"}
SOURCE_TIMESTAMP = re.compile(
    r"(?:^|_)(?:bg|pipe)_(\d+(?:\.\d+)?)_x\d+(?:_|$)", re.IGNORECASE
)


def source_timestamp(image_path: Path) -> str | None:
    """Return the source-frame timestamp encoded by a DRISHTI crop filename."""
    match = SOURCE_TIMESTAMP.search(image_path.stem)
    return match.group(1) if match else None


def external_validation_timestamps(external_validation_dir: Path) -> set[str]:
    """Collect source timestamps represented in the immutable validation set."""
    return {
        timestamp
        for path in external_validation_dir.iterdir()
        if path.is_file() and (timestamp := source_timestamp(path)) is not None
    }


def remap_pipeline_labels(contents: str) -> str:
    """Validate Pipeline-only source labels and remap source class 1 to class 0.

    An empty label file, malformed YOLO line, or any class other than 1 is not
    eligible for the Pipeline-only fine-tuning dataset.
    """
    remapped: list[str] = []
    for line_number, raw_line in enumerate(contents.splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric YOLO value") from exc
        if class_id != 1:
            raise ValueError(f"line {line_number}: expected only source class 1")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"line {line_number}: non-finite YOLO coordinate")
        remapped.append("0 " + " ".join(fields[1:]))
    if not remapped:
        raise ValueError("label file contains no Pipeline class 1 annotations")
    return "\n".join(remapped) + "\n"


def negative_source_classes(contents: str) -> list[int]:
    """Validate a pure non-Pipeline label file and return its source classes.

    Negatives are deliberately restricted to annotations from the known
    non-Pipeline DRISHTI classes.  In particular, a file that also contains a
    Pipeline annotation is not a negative candidate.
    """
    class_ids: list[int] = []
    for line_number, raw_line in enumerate(contents.splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric YOLO value") from exc
        if class_id not in {2, 3, 4}:
            raise ValueError(
                f"line {line_number}: expected only non-Pipeline source classes 2, 3, or 4"
            )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"line {line_number}: non-finite YOLO coordinate")
        class_ids.append(class_id)
    if not class_ids:
        raise ValueError("label file contains no non-Pipeline annotations")
    return class_ids


def split_timestamp_groups(timestamps: set[str], seed: int = SEED) -> dict[str, str]:
    """Deterministically assign whole source timestamps to an 80/20 split."""
    ordered = sorted(timestamps)
    random.Random(seed).shuffle(ordered)
    train_count = int(len(ordered) * 0.8)
    return {
        timestamp: "train" if index < train_count else "val"
        for index, timestamp in enumerate(ordered)
    }


def _validate_inputs(train_root: Path, external_validation_dir: Path) -> None:
    required = (train_root / "images", train_root / "labels", external_validation_dir)
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Required directory/directories not found: " + ", ".join(missing))


def _write_data_yaml(output_root: Path) -> None:
    (output_root / "data.yaml").write_text(
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: Pipeline\n",
        encoding="utf-8",
    )


def prepare_drishti_pipeline_finetune(
    train_root: Path = DRISHTI_TRAIN_DEFAULT,
    external_validation_dir: Path = EXTERNAL_VALIDATION_DEFAULT,
    output_root: Path = OUTPUT_DEFAULT,
    seed: int = SEED,
    include_negatives: bool = False,
) -> dict[str, Any]:
    """Create the derived dataset without changing either source dataset.

    Refuses to write into a non-empty output directory, preventing accidental
    mixture with a result built using different source data or a split seed.
    """
    train_root, external_validation_dir, output_root = map(
        Path, (train_root, external_validation_dir, output_root)
    )
    _validate_inputs(train_root, external_validation_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output_root}")

    validation_timestamps = external_validation_timestamps(external_validation_dir)
    candidates: list[tuple[Path, str, str, str, list[int]]] = []
    excluded: list[dict[str, str]] = []
    for image_path in sorted((train_root / "images").iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        timestamp = source_timestamp(image_path)
        if timestamp is None:
            excluded.append({"image": image_path.name, "reason": "missing_source_timestamp"})
            continue
        if timestamp in validation_timestamps:
            excluded.append({"image": image_path.name, "reason": "external_validation_timestamp_overlap"})
            continue
        label_path = train_root / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            excluded.append({"image": image_path.name, "reason": "missing_label"})
            continue
        try:
            labels = remap_pipeline_labels(label_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            if not include_negatives:
                excluded.append({"image": image_path.name, "reason": "not_pipeline_only", "detail": str(exc)})
                continue
            try:
                source_classes = negative_source_classes(
                    label_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, ValueError) as negative_exc:
                excluded.append(
                    {
                        "image": image_path.name,
                        "reason": "not_pipeline_positive_or_pure_non_pipeline_negative",
                        "detail": str(negative_exc),
                    }
                )
                continue
            candidates.append((image_path, timestamp, "", "negative", source_classes))
            continue
        candidates.append((image_path, timestamp, labels, "positive", [1]))

    assignments = split_timestamp_groups({timestamp for _, timestamp, _, _, _ in candidates}, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for image_path, timestamp, labels, image_type, source_classes in candidates:
        split = assignments[timestamp]
        image_destination = output_root / "images" / split / image_path.name
        label_destination = output_root / "labels" / split / f"{image_path.stem}.txt"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_destination)
        label_destination.write_text(labels, encoding="utf-8")
        record: dict[str, Any] = {
            "image": image_path.name,
            "source_timestamp": timestamp,
            "split": split,
        }
        if include_negatives:
            record.update({"type": image_type, "source_class_ids": source_classes})
        records.append(record)
    _write_data_yaml(output_root)

    split_counts = Counter(record["split"] for record in records)
    type_split_counts = {
        image_type: {
            split: sum(
                candidate_type == image_type and assignments[timestamp] == split
                for _, timestamp, _, candidate_type, _ in candidates
            )
            for split in ("train", "val")
        }
        for image_type in ("positive", "negative")
    }
    negative_source_class_counts = {
        split: {
            str(class_id): sum(
                source_classes.count(class_id)
                for _, timestamp, _, image_type, source_classes in candidates
                if assignments[timestamp] == split and image_type == "negative"
            )
            for class_id in (2, 3, 4)
        }
        for split in ("train", "val")
    }
    label_rule = (
        "Included only non-empty, well-formed labels whose every annotation uses source "
        "class 1; remapped source class 1 to class 0."
    )
    if include_negatives:
        label_rule += (
            " Pure non-Pipeline labels containing only source classes 2, 3, or 4 "
            "were also included as empty-label negatives."
        )
    manifest = {
        "dataset": "drishti_pipeline_finetune",
        "classes": {"0": "Pipeline"},
        "seed": seed,
        "rules": {
            "provenance": "Excluded training crops with unparseable source timestamps or source timestamps represented in the external validation set.",
            "labels": label_rule,
            "splits": "Unique source timestamps were shuffled deterministically with seed 42 and assigned 80/20 train/val; crops sharing a timestamp stay together.",
        },
        "external_validation_source_timestamp_count": len(validation_timestamps),
        "split_counts": {split: split_counts.get(split, 0) for split in ("train", "val")},
        "image_type_split_counts": type_split_counts,
        "negative_source_class_counts": negative_source_class_counts,
        "timestamp_group_counts": {
            split: sum(assignment == split for assignment in assignments.values())
            for split in ("train", "val")
        },
        "images": records,
        "excluded_images": excluded,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DRISHTI_TRAIN_DEFAULT)
    parser.add_argument("--external-validation-dir", type=Path, default=EXTERNAL_VALIDATION_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Include pure non-Pipeline DRISHTI crops as empty-label negatives.",
    )
    args = parser.parse_args()
    manifest = prepare_drishti_pipeline_finetune(**vars(args))
    print(json.dumps({"output": str(args.output_root), "split_counts": manifest["split_counts"], "excluded_images": len(manifest["excluded_images"])}, indent=2))


if __name__ == "__main__":
    main()
