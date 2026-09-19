"""Read-only compatibility audit for SONARIS and DRISHTI YOLO detection data.

Run with ``python -m src.data.audit_multiclass_datasets`` from the repository
root.  The script writes nothing: it prints a JSON record followed by a compact
human-readable assessment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


SONARIS_DEFAULT = Path("data/raw/SubPipeMiniSSS")
DRISHTI_DEFAULT = Path("data/external/drishti/val")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pbm", ".pgm", ".png", ".ppm", ".tif", ".tiff"}
UNIFIED_CLASSES = {0: "Pipeline", 1: "Shipwreck", 2: "Ghost Net", 3: "Mine Cylinder"}
DRISHTI_TO_UNIFIED = {1: 0, 2: 1, 3: 2, 4: 3}
DRISHTI_SOURCE_FRAME = re.compile(r"(?:^|_)bg_(\d+\.\d+)_x\d+$", re.IGNORECASE)
EXAMPLE_LIMIT = 20


def _files(directory: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _duplicate_values(values: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for value, name in values:
        grouped[value].append(name)
    return {value: names for value, names in grouped.items() if len(names) > 1}


def _examples(values: Iterable[str]) -> dict[str, Any]:
    values = list(values)
    return {"count": len(values), "examples": values[:EXAMPLE_LIMIT]}


def _duplicate_summary(duplicates: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "group_count": len(duplicates),
        "examples": {key: value for key, value in list(duplicates.items())[:EXAMPLE_LIMIT]},
    }


def _sampled_sha256(path: Path) -> str:
    """Fast duplicate-content screen: size plus the first/last 64 KiB."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(64 * 1024))
        if size > 64 * 1024:
            handle.seek(max(0, size - 64 * 1024))
            digest.update(handle.read(64 * 1024))
    return digest.hexdigest()


def _image_details(images: list[Path], root: Path) -> tuple[dict[str, int], list[str], dict[str, list[str]], dict[str, list[str]]]:
    dimensions: Counter[str] = Counter()
    unreadable: list[str] = []
    byte_hashes: list[tuple[str, str]] = []
    for path in images:
        name = path.relative_to(root).as_posix()
        byte_hashes.append((_sampled_sha256(path), name))
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable.append(name)
            continue
    return dict(sorted(dimensions.items())), unreadable, _duplicate_values(byte_hashes), {}


def _parse_yolo(path: Path) -> tuple[list[int], list[str], int]:
    """Return class IDs, errors, and non-empty line count for one YOLO file."""
    class_ids: list[int] = []
    errors: list[str] = []
    lines = 0
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return class_ids, ["not UTF-8"], lines
    for line_number, raw_line in enumerate(contents.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        lines += 1
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric class ID or coordinate")
            continue
        if class_id < 0 or any(not 0 < value <= 1 for value in values):
            errors.append(f"line {line_number}: class ID must be non-negative and coordinates must be in (0, 1]")
            continue
        class_ids.append(class_id)
    return class_ids, errors, lines


def _coco_classes(sonaris_root: Path) -> tuple[dict[int, str], dict[str, int], list[str]]:
    classes: dict[int, str] = {}
    boxes: Counter[str] = Counter()
    errors: list[str] = []
    for coco_path in sorted(sonaris_root.glob("DATA/SSS_*_images/COCO_Annotation/coco_format.json")):
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{coco_path}: {exc}")
            continue
        local_classes = {item.get("id"): item.get("name") for item in coco.get("categories", [])}
        for class_id, name in local_classes.items():
            if not isinstance(class_id, int) or not isinstance(name, str):
                errors.append(f"{coco_path}: malformed category")
            elif class_id in classes and classes[class_id] != name:
                errors.append(f"{coco_path}: conflicting name for COCO class {class_id}")
            else:
                classes[class_id] = name
        for annotation in coco.get("annotations", []):
            category_id = annotation.get("category_id")
            if category_id in local_classes:
                boxes[f"{category_id}:{local_classes[category_id]}"] += 1
            else:
                errors.append(f"{coco_path}: annotation has unknown category {category_id}")
    return dict(sorted(classes.items())), dict(sorted(boxes.items())), errors


def _audit_yolo_dataset(name: str, image_dirs: list[Path], label_dirs: list[Path], content_root: Path) -> dict[str, Any]:
    images = [path for directory in image_dirs if directory.exists() for path in _files(directory, IMAGE_EXTENSIONS)]
    labels = [path for directory in label_dirs if directory.exists() for path in _files(directory, {".txt"}) if path.name != "classes.txt"]
    # Include the immediate dataset group (e.g. SSS_HF_images) in the key so
    # identically timestamped HF and LF files are not mistaken for one image.
    image_stems = Counter(f"{path.parent.parent.name}/{path.stem}" for path in images)
    label_stems = Counter(f"{path.parent.parent.name}/{path.stem}" for path in labels)
    class_distribution: Counter[int] = Counter()
    invalid_labels: dict[str, list[str]] = {}
    total_lines = 0
    for label in labels:
        class_ids, errors, lines = _parse_yolo(label)
        total_lines += lines
        class_distribution.update(class_ids)
        if errors:
            invalid_labels[label.relative_to(content_root).as_posix()] = errors
    dimensions, unreadable, byte_duplicates, raster_duplicates = _image_details(images, content_root)
    missing_labels = sorted(stem for stem in image_stems if stem not in label_stems)
    orphan_labels = sorted(stem for stem in label_stems if stem not in image_stems)
    return {
        "name": name,
        "image_count": len(images),
        "label_file_count": len(labels),
        "image_dimensions": dimensions,
        "bounding_box_count": sum(class_distribution.values()),
        "yolo_non_empty_lines": total_lines,
        "class_ids_observed": sorted(class_distribution),
        "class_distribution": {str(key): value for key, value in sorted(class_distribution.items())},
        "missing_label_files": _examples(missing_labels),
        "orphan_label_files": _examples(orphan_labels),
        "invalid_label_files": {"count": len(invalid_labels), "examples": dict(list(invalid_labels.items())[:EXAMPLE_LIMIT])},
        "unreadable_images": _examples(unreadable),
        "duplicate_filenames": {
            "images": _duplicate_summary(_duplicate_values((path.name, path.relative_to(content_root).as_posix()) for path in images)),
            "labels": _duplicate_summary(_duplicate_values((path.name, path.relative_to(content_root).as_posix()) for path in labels)),
        },
        "duplicate_image_content": {
            "same_size_and_sampled_file_bytes": _duplicate_summary(byte_duplicates),
            "note": "This is a fast screen (first/last 64 KiB plus file size), not a perceptual-similarity test.",
        },
        "yolo_format_consistent": not invalid_labels,
    }


def audit(sonaris_root: Path = SONARIS_DEFAULT, drishti_root: Path = DRISHTI_DEFAULT) -> dict[str, Any]:
    """Audit the two supplied roots without changing either dataset."""
    sonaris_root, drishti_root = Path(sonaris_root), Path(drishti_root)
    if not sonaris_root.exists() or not drishti_root.exists():
        missing = [str(path) for path in (sonaris_root, drishti_root) if not path.exists()]
        raise FileNotFoundError("Dataset root(s) not found: " + ", ".join(missing))
    sonaris = _audit_yolo_dataset(
        "SONARIS", list(sonaris_root.glob("DATA/SSS_*_images/Image")),
        list(sonaris_root.glob("DATA/SSS_*_images/YOLO_Annotation")), sonaris_root,
    )
    coco_classes, coco_distribution, coco_errors = _coco_classes(sonaris_root)
    sonaris["coco_source_of_truth"] = {
        "class_ids_and_names": {str(key): value for key, value in coco_classes.items()},
        "bounding_box_distribution": coco_distribution,
        "annotation_errors": coco_errors,
        "yolo_class_mapping_by_classes_txt_order": {"0": "Pipeline"},
    }
    drishti = _audit_yolo_dataset("DRISHTI validation", [drishti_root / "images"], [drishti_root / "labels"], drishti_root)
    drishti_images = _files(drishti_root / "images", IMAGE_EXTENSIONS)
    sonaris_stems = {path.stem for path in _files(sonaris_root / "DATA", IMAGE_EXTENSIONS)}
    source_frame_matches = sorted(
        image.name for image in drishti_images
        if (match := DRISHTI_SOURCE_FRAME.search(image.stem)) and match.group(1) in sonaris_stems
    )
    unmapped = [class_id for class_id in drishti["class_ids_observed"] if class_id not in DRISHTI_TO_UNIFIED]
    drishti["proposed_unified_mapping"] = {
        "mapping": {str(key): value for key, value in UNIFIED_CLASSES.items()},
        "drishti_source_id_to_unified_id": {
            str(source_id): {
                "unified_id": unified_id,
                "class_name": UNIFIED_CLASSES[unified_id],
            }
            for source_id, unified_id in DRISHTI_TO_UNIFIED.items()
        },
        "observed_id_mapping": {
            str(source_id): UNIFIED_CLASSES[DRISHTI_TO_UNIFIED[source_id]]
            for source_id in drishti["class_ids_observed"] if source_id in DRISHTI_TO_UNIFIED
        },
        "unmapped_observed_ids": unmapped,
        "verification": "The validation root has no class-name manifest; this one-based source-ID mapping is the proposed mapping supplied for the audit and must be retained when preparing data.",
    }
    risks = []
    if source_frame_matches:
        risks.append(f"HIGH: {len(source_frame_matches)} DRISHTI images name SONARIS source-frame timestamps; treat DRISHTI val as non-independent until provenance/split overlap is resolved.")
    if drishti["missing_label_files"]["count"] or drishti["orphan_label_files"]["count"] or drishti["invalid_label_files"]["count"]:
        risks.append("HIGH: DRISHTI image/label pairing or YOLO formatting is incomplete; do not train on it unchanged.")
    if unmapped:
        risks.append(f"HIGH: DRISHTI contains IDs outside proposed unified mapping: {unmapped}.")
    if set(sonaris["class_ids_observed"]) != {0} or coco_classes != {1: "Pipeline"}:
        risks.append("HIGH: SONARIS YOLO/COCO class mapping differs from the expected Pipeline-only source truth.")
    if sonaris["missing_label_files"]["count"] or sonaris["orphan_label_files"]["count"]:
        risks.append("MEDIUM: SONARIS has images without label files and/or orphan labels; confirm these are intentional background frames before treating absent labels as negatives.")
    if sum(coco_distribution.values()) != sonaris["bounding_box_count"]:
        risks.append("HIGH: SONARIS COCO and YOLO bounding-box totals differ; resolve annotation-source disagreement before multiclass preparation.")
    if sonaris["duplicate_filenames"]["images"]["group_count"]:
        risks.append("MEDIUM: SONARIS HF and LF images share filenames; retain band/source paths when creating splits or derived filenames to prevent collisions.")
    if drishti["duplicate_image_content"]["same_size_and_sampled_file_bytes"]["group_count"]:
        risks.append("MEDIUM: DRISHTI has candidate duplicate image-content groups; inspect them before splitting or scoring validation data.")
    risks.append("MEDIUM: SONARIS uses wide 5000x500 frames while DRISHTI may use crops/resized imagery; normalize preprocessing and split by source acquisition, not by image file.")
    return {
        "audit": "multiclass_dataset_compatibility",
        "sources": {"sonaris": str(sonaris_root), "drishti_validation": str(drishti_root)},
        "proposed_unified_mapping": {str(key): value for key, value in UNIFIED_CLASSES.items()},
        "sonaris": sonaris,
        "drishti": drishti,
        "cross_dataset": {
            "matching_filenames": _examples(sorted(set(path.name for path in _files(sonaris_root / "DATA", IMAGE_EXTENSIONS)) & set(path.name for path in drishti_images))),
            "drishti_images_matching_sonaris_source_timestamp": _examples(source_frame_matches),
        },
        "compatibility_and_leakage_risks": risks,
    }


def _human_summary(report: dict[str, Any]) -> str:
    sonaris, drishti = report["sonaris"], report["drishti"]
    lines = [
        "Dataset compatibility audit",
        f"SONARIS: {sonaris['image_count']} images, {sonaris['label_file_count']} YOLO labels, {sonaris['bounding_box_count']} valid YOLO boxes; COCO classes {sonaris['coco_source_of_truth']['class_ids_and_names']}.",
        f"DRISHTI val: {drishti['image_count']} images, {drishti['label_file_count']} labels, {drishti['bounding_box_count']} valid YOLO boxes; observed IDs {drishti['class_ids_observed']}.",
        f"Format issues: SONARIS={sonaris['invalid_label_files']['count']}, DRISHTI={drishti['invalid_label_files']['count']}; DRISHTI missing/orphan labels={drishti['missing_label_files']['count']}/{drishti['orphan_label_files']['count']}.",
        f"Potential DRISHTI-to-SONARIS source-frame overlaps: {report['cross_dataset']['drishti_images_matching_sonaris_source_timestamp']['count']}.",
        "Risks:",
    ]
    lines.extend(f"- {risk}" for risk in report["compatibility_and_leakage_risks"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonaris-root", type=Path, default=SONARIS_DEFAULT)
    parser.add_argument("--drishti-root", type=Path, default=DRISHTI_DEFAULT)
    args = parser.parse_args()
    report = audit(args.sonaris_root, args.drishti_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    print(_human_summary(report))


if __name__ == "__main__":
    main()
