"""Build a new, auditable SONARIS/DRISHTI multiclass derived dataset.

Raw inputs are read only.  The output directory must not already contain data;
this prevents accidental replacement of a previous derived dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.compare_sonar_annotations import (
    PIPELINE_COCO_ID,
    PIPELINE_YOLO_ID,
    _match_boxes,
    _parse_yolo as parse_pipeline_yolo,
    coco_to_yolo,
)


SONARIS_RAW_DEFAULT = Path("data/raw/SubPipeMiniSSS/DATA")
DRISHTI_DEFAULT = Path("data/external/drishti/val")
PROVENANCE_DEFAULT = Path("reports/dataset/provenance_report.json")
OUTPUT_DEFAULT = Path("data/processed/sonaris_multiclass")
CLASSES = {0: "Pipeline", 1: "Shipwreck", 2: "Ghost Net", 3: "Mine Cylinder"}
DRISHTI_REMAP = {1: 0, 2: 1, 3: 2, 4: 3}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pbm", ".pgm", ".png", ".ppm", ".tif", ".tiff"}


def unique_filename(source_dataset: str, band: str | None, original_name: str) -> str:
    """Create a collision-safe output name without losing source identity."""
    prefix = source_dataset.lower()
    if band:
        prefix += "_" + band.replace("SSS_", "").replace("_images", "").lower()
    return f"{prefix}_{original_name}"


def _parse_drishti_labels(contents: str) -> tuple[str, list[int], list[tuple[int, list[float]]]]:
    """Validate one DRISHTI label file and remap its one-based class IDs."""
    output: list[str] = []
    class_ids: list[int] = []
    boxes: list[tuple[int, list[float]]] = []
    for line_number, raw_line in enumerate(contents.splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected 5 YOLO fields")
        try:
            source_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric YOLO value") from exc
        if (
            source_id not in DRISHTI_REMAP
            or any(not math.isfinite(value) for value in coordinates)
            or coordinates[2] <= 0
            or coordinates[3] <= 0
        ):
            raise ValueError(f"line {line_number}: invalid DRISHTI class ID or normalized coordinate")
        unified_id = DRISHTI_REMAP[source_id]
        class_ids.append(unified_id)
        boxes.append((unified_id, coordinates))
        output.append(f"{unified_id} " + " ".join(fields[1:]))
    return ("\n".join(output) + ("\n" if output else ""), class_ids, boxes)


def remap_drishti_labels(contents: str) -> tuple[str, list[int]]:
    """Validate one DRISHTI label file and remap its one-based class IDs."""
    labels, class_ids, _ = _parse_drishti_labels(contents)
    return labels, class_ids


def _out_of_bounds_drishti_bbox(
    boxes: list[tuple[int, list[float]]], image_width: int, image_height: int
) -> tuple[int, list[float]] | None:
    """Return the first remapped YOLO bbox extending outside its source image."""
    for unified_class_id, original_bbox in boxes:
        cx, cy, width, height = original_bbox
        x1 = (cx - width / 2) * image_width
        y1 = (cy - height / 2) * image_height
        x2 = (cx + width / 2) * image_width
        y2 = (cy + height / 2) * image_height
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            return unified_class_id, original_bbox
    return None


def _write_record(output_root: Path, record: dict[str, Any], image_source: Path, label_contents: str) -> None:
    split = record["split"]
    image_destination = output_root / "images" / split / record["derived_filename"]
    label_destination = output_root / "labels" / split / f"{Path(record['derived_filename']).stem}.txt"
    image_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_source, image_destination)
    label_destination.write_text(label_contents, encoding="utf-8")


def _sonaris_records(raw_root: Path) -> tuple[list[tuple[dict[str, Any], Path, str]], list[dict[str, Any]]]:
    included: list[tuple[dict[str, Any], Path, str]] = []
    excluded: list[dict[str, Any]] = []
    for band in ("SSS_HF_images", "SSS_LF_images"):
        band_root = raw_root / band
        images = {path.stem: path for path in (band_root / "Image").iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in (band_root / "YOLO_Annotation").glob("*.txt") if path.name != "classes.txt"}
        coco = json.loads((band_root / "COCO_Annotation" / "coco_format.json").read_text(encoding="utf-8"))
        coco_images = {Path(image["file_name"]).stem: image for image in coco.get("images", [])}
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            annotations[annotation.get("image_id")].append(annotation)
        for stem, image_path in sorted(images.items()):
            label_path, coco_image = labels.get(stem), coco_images.get(stem)
            if label_path is None:
                excluded.append({"source_dataset": "SONARIS", "band": band, "original_filename": image_path.name, "reason": "missing_yolo_label_not_assumed_negative"})
                continue
            if coco_image is None:
                excluded.append({"source_dataset": "SONARIS", "band": band, "original_filename": image_path.name, "reason": "yolo_label_has_no_matching_coco_image"})
                continue
            yolo_boxes, yolo_errors = parse_pipeline_yolo(label_path)
            raw_annotations = annotations.get(coco_image["id"], [])
            valid_annotations = [item for item in raw_annotations if item.get("category_id") == PIPELINE_COCO_ID and len(item.get("bbox", [])) == 4]
            coco_boxes = [coco_to_yolo(item["bbox"], coco_image["width"], coco_image["height"]) for item in valid_annotations]
            comparison = _match_boxes(coco_boxes, yolo_boxes)
            verified = not yolo_errors and len(raw_annotations) == len(valid_annotations) and len(coco_boxes) == len(yolo_boxes) and comparison["exact"] == len(coco_boxes)
            if not verified or not yolo_boxes:
                excluded.append({"source_dataset": "SONARIS", "band": band, "original_filename": image_path.name, "reason": "coco_yolo_not_verified_consistent_or_empty", "coco_box_count": len(coco_boxes), "yolo_box_count": len(yolo_boxes)})
                continue
            record = {
                "source_dataset": "SONARIS",
                "band": band,
                "original_filename": image_path.name,
                "derived_filename": unique_filename("sonaris", band, image_path.name),
                "unified_class_ids": [PIPELINE_YOLO_ID],
                "box_count": len(yolo_boxes),
                "split_group": stem,
                "verification": "exact COCO-to-YOLO match",
            }
            included.append((record, image_path, label_path.read_text(encoding="utf-8")))
        for stem, label_path in sorted(labels.items()):
            if stem not in images:
                excluded.append({"source_dataset": "SONARIS", "band": band, "original_filename": label_path.name, "reason": "orphan_yolo_label_without_source_image"})
    return included, excluded


def _assign_sonaris_splits(records: list[tuple[dict[str, Any], Path, str]]) -> None:
    """Chronological 80/10/10 split, keeping same-timestamp HF/LF frames together."""
    groups = sorted({record[0]["split_group"] for record in records}, key=lambda value: (float(value) if value.replace(".", "", 1).isdigit() else float("inf"), value))
    train_cutoff, val_cutoff = int(len(groups) * 0.8), int(len(groups) * 0.9)
    assignments = {group: "train" if index < train_cutoff else "val" if index < val_cutoff else "test" for index, group in enumerate(groups)}
    for record, _, _ in records:
        record["split"] = assignments[record["split_group"]]


def _provenance_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Provenance report not found: {path}. Run check_dataset_provenance first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["drishti_image"]: item["status"] for item in report.get("relationships", [])}


def external_validation_allowed(status: str | None) -> bool:
    """Only explicit provenance leakage is barred from the external manifest."""
    return status != "LEAKAGE_RISK"


def _drishti_records(drishti_root: Path, statuses: dict[str, str], include_for_training: bool) -> tuple[list[tuple[dict[str, Any], Path, str]], list[dict[str, Any]]]:
    included: list[tuple[dict[str, Any], Path, str]] = []
    excluded: list[dict[str, Any]] = []
    for image_path in sorted((drishti_root / "images").iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_image = f"images/{image_path.name}"
        label_path = drishti_root / "labels" / f"{image_path.stem}.txt"
        status = statuses.get(relative_image, "NO_SONARIS_TIMESTAMP_REFERENCE")
        if not label_path.exists():
            excluded.append({"source_dataset": "DRISHTI", "original_filename": image_path.name, "reason": "missing_yolo_label"})
            continue
        try:
            labels, class_ids, boxes = _parse_drishti_labels(label_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            excluded.append({"source_dataset": "DRISHTI", "original_filename": image_path.name, "reason": "invalid_yolo_label", "detail": str(exc)})
            continue
        if include_for_training:
            included.append(({
                "source_dataset": "DRISHTI", "original_filename": image_path.name,
                "derived_filename": unique_filename("drishti", None, image_path.name), "unified_class_ids": sorted(set(class_ids)),
                "box_count": len(class_ids), "split": "train", "provenance_status": status,
            }, image_path, labels))
        elif external_validation_allowed(status):
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            invalid_bbox = _out_of_bounds_drishti_bbox(boxes, image_width, image_height)
            if invalid_bbox is not None:
                unified_class_id, original_bbox = invalid_bbox
                excluded.append({
                    "original_filename": image_path.name,
                    "source_dataset": "DRISHTI",
                    "split": "external_validation",
                    "unified_class_id": unified_class_id,
                    "original_bbox": original_bbox,
                    "image_width": image_width,
                    "image_height": image_height,
                    "reason": "OUT_OF_BOUNDS_BBOX",
                })
                continue
            included.append(({
                "source_dataset": "DRISHTI", "original_filename": image_path.name,
                "derived_filename": unique_filename("drishti", None, image_path.name), "unified_class_ids": sorted(set(class_ids)),
                "box_count": len(class_ids), "split": "external_validation", "provenance_status": status,
            }, image_path, labels))
        else:
            excluded.append({"source_dataset": "DRISHTI", "original_filename": image_path.name, "reason": "provenance_leakage_risk_excluded_from_external_validation", "provenance_status": status})
    return included, excluded


def prepare_multiclass_dataset(
    sonaris_raw_root: Path = SONARIS_RAW_DEFAULT,
    drishti_root: Path = DRISHTI_DEFAULT,
    provenance_report: Path = PROVENANCE_DEFAULT,
    output_root: Path = OUTPUT_DEFAULT,
    include_drishti_train: bool = False,
) -> dict[str, Any]:
    """Create the derived dataset, refusing to replace a non-empty output root."""
    sonaris_raw_root, drishti_root, provenance_report, output_root = map(Path, (sonaris_raw_root, drishti_root, provenance_report, output_root))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output_root}")
    statuses = _provenance_statuses(provenance_report)
    sonaris_records, excluded = _sonaris_records(sonaris_raw_root)
    _assign_sonaris_splits(sonaris_records)
    drishti_records, drishti_excluded = _drishti_records(drishti_root, statuses, include_drishti_train)
    excluded.extend(drishti_excluded)
    records = sonaris_records + drishti_records
    for record, image_path, labels in records:
        _write_record(output_root, record, image_path, labels)
    manifest_records = [record for record, _, _ in records]
    # Count per-box labels accurately when images contain multiple classes.
    class_distribution = Counter()
    for record, _, labels in records:
        for line in labels.splitlines():
            if line.strip():
                class_distribution[int(line.split()[0])] += 1
    split_counts = Counter(record["split"] for record in manifest_records)
    manifest = {
        "dataset": "sonaris_multiclass",
        "classes": {str(key): value for key, value in CLASSES.items()},
        "rules": {
            "sonaris": "Only exact COCO-to-YOLO-consistent, positive Pipeline labels are included; missing YOLO labels are not negatives; orphan labels are excluded.",
            "drishti": "Source IDs 1/2/3/4 are remapped to unified 0/1/2/3. Training inclusion requires --include-drishti-train.",
            "external_validation": "Stored in images/external_validation and labels/external_validation; LEAKAGE_RISK images and DRISHTI boxes extending outside source-image bounds are excluded.",
            "splits": "SONARIS uses chronological 80/10/10 timestamp groups with HF/LF same-timestamp frames kept together.",
        },
        "include_drishti_train": include_drishti_train,
        "class_distribution": {str(key): value for key, value in sorted(class_distribution.items())},
        "split_counts": dict(sorted(split_counts.items())),
        "images": manifest_records,
        "excluded_images": excluded,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonaris-raw-root", type=Path, default=SONARIS_RAW_DEFAULT)
    parser.add_argument("--drishti-root", type=Path, default=DRISHTI_DEFAULT)
    parser.add_argument("--provenance-report", type=Path, default=PROVENANCE_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--include-drishti-train", action="store_true", help="Explicitly include remapped DRISHTI data in train only.")
    args = parser.parse_args()
    manifest = prepare_multiclass_dataset(**vars(args))
    print(json.dumps({"output": str(args.output_root), "split_counts": manifest["split_counts"], "class_distribution": manifest["class_distribution"], "excluded_images": len(manifest["excluded_images"])}, indent=2))


if __name__ == "__main__":
    main()
