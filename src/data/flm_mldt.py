"""Validate and prepare the external FLM-MLDT sonar-debris dataset.

This module is deliberately read-only with respect to the Downloads source.  A
validation pass writes JSON only; a build pass derives a new YOLO directory.
Unknown class IDs stop builds by default so no label is silently changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

CLASSES = ["Aluminum_tube", "Fish_net", "Floating_floor_wood", "Plastic_Deck", "PVC_cone", "PVC_perforated_deck", "PVC_Square", "Vinyl", "Wooden_deck", "PVC_Blue_Square", "rope"]
DEFAULT_IMAGES = Path(r"C:\Users\mahes\Downloads\polar")
DEFAULT_ANNOTATIONS = Path(r"C:\Users\mahes\Downloads\annotations_navigation\annotations")
DEFAULT_OUTPUT = Path("data/processed/debris_detection")


@dataclass(frozen=True)
class Record:
    sequence: str
    image: Path
    label: Path


def _sequence(path: Path, root: Path) -> str:
    return path.relative_to(root).parent.as_posix()


def _label_index(path: Path) -> int | None:
    try:
        return int(path.name.split("_", 1)[0])
    except ValueError:
        return None


def _parse(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[dict]]:
    rows, errors = [], []
    for line_no, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        fields = text.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except (ValueError, IndexError):
            errors.append({"line": line_no, "reason": "not five numeric YOLO fields"})
            continue
        if len(fields) != 5 or len(values) != 4:
            errors.append({"line": line_no, "reason": "not five YOLO fields"})
            continue
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1 and x - width / 2 >= 0 and x + width / 2 <= 1 and y - height / 2 >= 0 and y + height / 2 <= 1):
            errors.append({"line": line_no, "reason": "box outside normalized image bounds"})
            continue
        rows.append((class_id, x, y, width, height))
    return rows, errors


def discover(images_root: Path, annotations_root: Path) -> tuple[list[Record], list[str], list[str]]:
    """Match labels to image frames by their documented numerical frame index."""
    records, missing_images, missing_labels = [], [], []
    image_sequences = {path.relative_to(images_root).parent for path in images_root.rglob("*.png")}
    label_sequences = {path.relative_to(annotations_root).parent for path in annotations_root.rglob("*.txt")}
    for relative_sequence in sorted(image_sequences | label_sequences, key=lambda value: value.as_posix()):
        images = sorted((images_root / relative_sequence).glob("*.png"))
        labels = sorted((annotations_root / relative_sequence).glob("*.txt"), key=lambda path: (_label_index(path) is None, _label_index(path), path.name))
        by_index = {_label_index(path): path for path in labels if _label_index(path) is not None}
        for index, image in enumerate(images):
            label = by_index.get(index)
            if label is None:
                missing_labels.append(image.relative_to(images_root).as_posix())
            else:
                records.append(Record(relative_sequence.as_posix(), image, label))
        for index, label in by_index.items():
            if index >= len(images):
                missing_images.append(label.relative_to(annotations_root).as_posix())
    return records, missing_images, missing_labels


def validate(images_root: Path = DEFAULT_IMAGES, annotations_root: Path = DEFAULT_ANNOTATIONS) -> dict:
    images_root, annotations_root = Path(images_root), Path(annotations_root)
    records, missing_images, missing_labels = discover(images_root, annotations_root)
    total_images = len(list(images_root.rglob("*.png")))
    label_files = list(annotations_root.rglob("*.txt"))
    distribution, sequences, invalid_ids, malformed, empty = Counter(), Counter(), defaultdict(list), [], 0
    for record in records:
        rows, errors = _parse(record.label)
        sequences[record.sequence] += 1
        if not rows and not errors:
            empty += 1
        malformed.extend({"file": record.label.relative_to(annotations_root).as_posix(), **error} for error in errors)
        for class_id, *_ in rows:
            distribution[class_id] += 1
            if class_id not in range(len(CLASSES)):
                invalid_ids[class_id].append(record.label.relative_to(annotations_root).as_posix())
    return {
        "source": {"images_root": str(images_root), "annotations_root": str(annotations_root)},
        "official_classes": {str(i): name for i, name in enumerate(CLASSES)},
        "total_images": total_images, "total_annotation_files": len(label_files),
        "matched_pairs": len(records), "empty_annotations": empty, "non_empty_annotations": len(records) - empty,
        "missing_images_for_labels": missing_images, "missing_labels_for_images": missing_labels,
        "invalid_class_ids": {str(key): value for key, value in sorted(invalid_ids.items())},
        "class_distribution": {str(key): value for key, value in sorted(distribution.items())},
        "sequence_distribution": dict(sorted(sequences.items())), "malformed_or_out_of_bounds_rows": malformed,
        "class_11_investigation": {
            "status": "unresolved", "files": invalid_ids.get(11, []),
            "sequences": sorted({_sequence(annotations_root / file, annotations_root) for file in invalid_ids.get(11, [])}),
            "evidence": "data_config.yaml declares nc: 11 and names IDs 0-10 only; no class-11 name is present.",
            "handling": "The builder refuses unknown IDs by default. It never deletes or remaps a class-11 row.",
        },
    }


def _split(sequence: str, seed: str) -> str:
    # Entire sequences stay together; deterministic hash avoids frame leakage.
    bucket = int(hashlib.sha256((seed + sequence).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def build(report: dict, output: Path = DEFAULT_OUTPUT, seed: str = "sonaris-flm-mldt-v1", unknown_policy: str = "fail", link_mode: str = "auto") -> dict:
    invalid_files = {file for files in report["invalid_class_ids"].values() for file in files}
    invalid_sequences = {_sequence(Path(report["source"]["annotations_root"]) / file, Path(report["source"]["annotations_root"])) for file in invalid_files}
    if invalid_files and unknown_policy == "fail":
        raise ValueError("Unknown class IDs found; inspect the validation report or use --unknown-policy exclude-sequences.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists: {output}. Choose an empty directory.")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    records, _, _ = discover(Path(report["source"]["images_root"]), Path(report["source"]["annotations_root"]))
    written, excluded, split_counts = 0, 0, Counter()
    for record in records:
        if record.sequence in invalid_sequences and unknown_policy == "exclude-sequences":
            excluded += 1; continue
        split = _split(record.sequence, seed)
        name = record.sequence.replace("/", "__") + "__" + record.image.name
        destination = output / "images" / split / name
        if link_mode in ("auto", "hardlink"):
            try: os.link(record.image, destination)
            except OSError:
                if link_mode == "hardlink": raise
                shutil.copy2(record.image, destination)
        else: shutil.copy2(record.image, destination)
        shutil.copy2(record.label, output / "labels" / split / (Path(name).stem + ".txt"))
        written += 1; split_counts[split] += 1
    (output / "data.yaml").write_text("path: " + output.resolve().as_posix() + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)), encoding="utf-8")
    summary = {"written_images": written, "excluded_images": excluded, "split_counts": dict(split_counts), "unknown_policy": unknown_policy, "link_mode": link_mode}
    (output / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "build"))
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--report", type=Path, default=Path("reports/flm_mldt_validation.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unknown-policy", choices=("fail", "exclude-sequences"), default="fail")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args(); report = validate(args.images_root, args.annotations_root)
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result = report if args.command == "validate" else build(report, args.output, unknown_policy=args.unknown_policy, link_mode=args.link_mode)
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
