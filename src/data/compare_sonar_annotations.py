"""Read-only COCO-to-YOLO annotation comparison for SONARIS HF and LF data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RAW_DEFAULT = Path("data/raw/SubPipeMiniSSS/DATA")
REPORT_DEFAULT = Path("reports/dataset/sonaris_annotation_comparison.json")
PIPELINE_COCO_ID = 1
PIPELINE_YOLO_ID = 0
EXACT_TOLERANCE = 1e-5
IOU_MATCH_THRESHOLD = 0.5
COORDINATE_CONVERSION_IOU = 0.99
EXAMPLE_LIMIT = 20


def coco_to_yolo(bbox: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    """Convert a COCO [x, y, width, height] box to normalized YOLO xywh."""
    x, y, box_width, box_height = bbox
    return ((x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height)


def yolo_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = box
    return (x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2)


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Return IoU for two normalized YOLO xywh boxes."""
    first_x1, first_y1, first_x2, first_y2 = yolo_to_xyxy(first)
    second_x1, second_y1, second_x2, second_y2 = yolo_to_xyxy(second)
    overlap_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    overlap_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = overlap_width * overlap_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _is_exact(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return all(abs(left - right) <= EXACT_TOLERANCE for left, right in zip(first, second))


def _duplicate_indices(boxes: list[tuple[float, float, float, float]]) -> list[int]:
    """Return every occurrence after the first equivalent box."""
    duplicates: list[int] = []
    unique: list[tuple[float, float, float, float]] = []
    for index, box in enumerate(boxes):
        if any(_is_exact(box, previous) for previous in unique):
            duplicates.append(index)
        else:
            unique.append(box)
    return duplicates


def _parse_yolo(path: Path) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    boxes: list[tuple[float, float, float, float]] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(f"line {line_number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            box = tuple(float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric values")
            continue
        if class_id != PIPELINE_YOLO_ID or any(value <= 0 or value > 1 for value in box):
            errors.append(f"line {line_number}: expected Pipeline class 0 and normalized coordinates in (0, 1]")
            continue
        boxes.append(box)  # type: ignore[arg-type]
    return boxes, errors


def _match_boxes(coco_boxes: list[tuple[float, float, float, float]], yolo_boxes: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    """Greedily form the maximum-IoU one-to-one assignment."""
    candidates = sorted(
        ((iou(coco_box, yolo_box), coco_index, yolo_index)
         for coco_index, coco_box in enumerate(coco_boxes)
         for yolo_index, yolo_box in enumerate(yolo_boxes)),
        reverse=True,
    )
    used_coco: set[int] = set()
    used_yolo: set[int] = set()
    exact = coordinate_conversion = mismatched = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for overlap, coco_index, yolo_index in candidates:
        if overlap < IOU_MATCH_THRESHOLD or coco_index in used_coco or yolo_index in used_yolo:
            continue
        used_coco.add(coco_index)
        used_yolo.add(yolo_index)
        item = {
            "coco_index": coco_index,
            "yolo_index": yolo_index,
            "iou": round(overlap, 8),
            "coco_normalized_yolo": coco_boxes[coco_index],
            "yolo": yolo_boxes[yolo_index],
        }
        if _is_exact(coco_boxes[coco_index], yolo_boxes[yolo_index]):
            exact += 1
        elif overlap >= COORDINATE_CONVERSION_IOU:
            coordinate_conversion += 1
            examples["coordinate_conversion"].append(item)
        else:
            mismatched += 1
            examples["mismatched"].append(item)
    missing = [index for index in range(len(coco_boxes)) if index not in used_coco]
    extra = [index for index in range(len(yolo_boxes)) if index not in used_yolo]
    return {
        "exact": exact,
        "coordinate_conversion": coordinate_conversion,
        "mismatched": mismatched,
        "missing_coco_indices": missing,
        "extra_yolo_indices": extra,
        "examples": {key: value[:EXAMPLE_LIMIT] for key, value in examples.items()},
    }


def _sample(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"count": len(items), "examples": items[:EXAMPLE_LIMIT]}


def _compare_band(raw_root: Path, band: str) -> dict[str, Any]:
    band_root = raw_root / band
    coco_path = band_root / "COCO_Annotation" / "coco_format.json"
    labels_root = band_root / "YOLO_Annotation"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {Path(image["file_name"]).stem: image for image in coco.get("images", [])}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations[annotation.get("image_id")].append(annotation)
    labels = {path.stem: path for path in labels_root.glob("*.txt") if path.name != "classes.txt"}
    totals: Counter[str] = Counter()
    mismatch_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shared_stems = sorted(set(images) & set(labels))
    for stem in shared_stems:
        image = images[stem]
        raw_annotations = annotations.get(image["id"], [])
        invalid_coco = [annotation for annotation in raw_annotations if annotation.get("category_id") != PIPELINE_COCO_ID or len(annotation.get("bbox", [])) != 4]
        coco_boxes = [
            coco_to_yolo(annotation["bbox"], image["width"], image["height"])
            for annotation in raw_annotations if annotation not in invalid_coco
        ]
        yolo_boxes, yolo_errors = _parse_yolo(labels[stem])
        match = _match_boxes(coco_boxes, yolo_boxes)
        coco_duplicates = _duplicate_indices(coco_boxes)
        yolo_duplicates = _duplicate_indices(yolo_boxes)
        totals.update({
            "shared_images": 1,
            "coco_boxes": len(coco_boxes),
            "yolo_boxes": len(yolo_boxes),
            "exact_boxes": match["exact"],
            "coordinate_conversion_boxes": match["coordinate_conversion"],
            "mismatched_boxes": match["mismatched"],
            "missing_yolo_boxes": len(match["missing_coco_indices"]),
            "extra_yolo_boxes": len(match["extra_yolo_indices"]),
            "invalid_coco_annotations": len(invalid_coco),
            "invalid_yolo_lines": len(yolo_errors),
            "duplicate_coco_annotations": len(coco_duplicates),
            "duplicate_yolo_annotations": len(yolo_duplicates),
        })
        if len(coco_boxes) != len(yolo_boxes):
            mismatch_examples["box_count_difference"].append({"image_stem": stem, "coco_box_count": len(coco_boxes), "yolo_box_count": len(yolo_boxes)})
        for kind in ("coordinate_conversion", "mismatched"):
            for item in match["examples"].get(kind, []):
                mismatch_examples[kind].append({"image_stem": stem, **item})
        for index in match["missing_coco_indices"]:
            mismatch_examples["missing_yolo"].append({"image_stem": stem, "coco_normalized_yolo": coco_boxes[index]})
        for index in match["extra_yolo_indices"]:
            mismatch_examples["extra_yolo"].append({"image_stem": stem, "yolo": yolo_boxes[index]})
        if invalid_coco:
            mismatch_examples["invalid_coco_annotation"].append({"image_stem": stem, "count": len(invalid_coco)})
        if yolo_errors:
            mismatch_examples["invalid_yolo_label"].append({"image_stem": stem, "errors": yolo_errors})
        for index in coco_duplicates:
            mismatch_examples["duplicate_coco_annotation"].append({"image_stem": stem, "coco_normalized_yolo": coco_boxes[index]})
        for index in yolo_duplicates:
            mismatch_examples["duplicate_yolo_annotation"].append({"image_stem": stem, "yolo": yolo_boxes[index]})
    coco_without_yolo = [{"image_stem": stem, "coco_annotation_count": len(annotations.get(image["id"], []))} for stem, image in images.items() if stem not in labels]
    yolo_without_coco_image = [{"image_stem": stem} for stem in labels if stem not in images]
    yolo_without_coco_annotation = [{"image_stem": stem} for stem in shared_stems if not annotations.get(images[stem]["id"], [])]
    totals.update({
        "coco_images": len(images), "yolo_label_files": len(labels),
        "coco_images_missing_yolo": len(coco_without_yolo),
        "yolo_labels_missing_coco_image": len(yolo_without_coco_image),
        "yolo_labels_with_coco_image_but_no_annotation": len(yolo_without_coco_annotation),
    })
    return {
        "band": band,
        "class_mapping": {"coco": {str(PIPELINE_COCO_ID): "Pipeline"}, "yolo": {str(PIPELINE_YOLO_ID): "Pipeline"}},
        "summary": dict(totals),
        "difference_diagnosis": {
            "coordinate_conversion_or_serialization": totals["coordinate_conversion_boxes"],
            "duplicate_coco_annotations": totals["duplicate_coco_annotations"],
            "duplicate_yolo_annotations": totals["duplicate_yolo_annotations"],
            "image_name_mapping": totals["coco_images_missing_yolo"] + totals["yolo_labels_missing_coco_image"],
            "genuinely_different_annotations": totals["mismatched_boxes"] + totals["missing_yolo_boxes"] + totals["extra_yolo_boxes"],
        },
        "coco_images_missing_yolo": _sample(coco_without_yolo),
        "yolo_labels_missing_coco_image": _sample(yolo_without_coco_image),
        "yolo_labels_with_coco_image_but_no_annotation": _sample(yolo_without_coco_annotation),
        "mismatch_examples": {kind: _sample(items) for kind, items in sorted(mismatch_examples.items())},
    }


def compare_annotations(raw_root: Path = RAW_DEFAULT) -> dict[str, Any]:
    raw_root = Path(raw_root)
    reports = [_compare_band(raw_root, band) for band in ("SSS_HF_images", "SSS_LF_images")]
    overall: Counter[str] = Counter()
    for report in reports:
        overall.update(report["summary"])
    return {
        "audit": "sonaris_coco_yolo_annotation_comparison",
        "source_root": str(raw_root),
        "matching": {"exact_coordinate_tolerance": EXACT_TOLERANCE, "iou_threshold": IOU_MATCH_THRESHOLD, "coordinate_conversion_iou": COORDINATE_CONVERSION_IOU},
        "overall_summary": dict(overall),
        "bands": reports,
    }


def _human_summary(report: dict[str, Any]) -> str:
    summary = report["overall_summary"]
    return "\n".join([
        "SONARIS COCO-to-YOLO annotation comparison",
        f"Shared images: {summary.get('shared_images', 0)} | COCO boxes: {summary.get('coco_boxes', 0)} | YOLO boxes: {summary.get('yolo_boxes', 0)}",
        f"Exact: {summary.get('exact_boxes', 0)} | conversion-level differences: {summary.get('coordinate_conversion_boxes', 0)} | mismatched: {summary.get('mismatched_boxes', 0)} | missing YOLO: {summary.get('missing_yolo_boxes', 0)} | extra YOLO: {summary.get('extra_yolo_boxes', 0)}",
        f"COCO images without YOLO: {summary.get('coco_images_missing_yolo', 0)} | YOLO labels without COCO image: {summary.get('yolo_labels_missing_coco_image', 0)} | YOLO labels with no COCO annotation: {summary.get('yolo_labels_with_coco_image_but_no_annotation', 0)}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--report-path", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    report = compare_annotations(args.raw_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(_human_summary(report))
    print(f"JSON report: {args.report_path}")


if __name__ == "__main__":
    main()
