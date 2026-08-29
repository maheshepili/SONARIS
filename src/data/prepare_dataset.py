"""Create a validated, derived YOLO dataset without changing raw sonar data."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from src.data.preprocess import preprocess_sonar_image


RAW_DEFAULT = Path("data/raw/SubPipeMiniSSS/DATA")
OUTPUT_DEFAULT = Path("data/processed/sonaris_detection")
EXPECTED_CLASS = "Pipeline"
IMAGE_EXTENSIONS = {".pbm", ".bpm", ".ppm", ".png", ".jpg", ".jpeg"}


@dataclass
class PreparationReport:
    source_root: str
    output_root: str
    images_written: int = 0
    train_images: int = 0
    val_images: int = 0
    labelled_images: int = 0
    annotation_rows: int = 0
    invalid_label_rows: int = 0
    orphan_label_files: int = 0
    unreadable_images: int = 0
    coco_invalid_boxes: int = 0
    split_cutoff: str = ""


def _valid_yolo_line(line: str) -> bool:
    fields = line.split()
    if len(fields) != 5 or fields[0] != "0":
        return False
    try:
        x, y, width, height = (float(value) for value in fields[1:])
    except ValueError:
        return False
    return 0 < x <= 1 and 0 < y <= 1 and 0 < width <= 1 and 0 < height <= 1


def _read_valid_labels(path: Path) -> tuple[list[str], int]:
    if not path.exists():
        return [], 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [line for line in lines if _valid_yolo_line(line)]
    return valid, len(lines) - len(valid)


def _timestamp_key(stem: str) -> tuple[int, float | str]:
    """Sort timestamp-named frames before non-standard names such as orphan data."""
    try:
        return 0, float(stem)
    except ValueError:
        return 1, stem


def _validate_coco(coco_path: Path) -> int:
    """Return truly invalid COCO boxes, allowing insignificant float rounding."""
    if not coco_path.exists():
        return 0
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in coco.get("images", [])}
    categories = {category["id"] for category in coco.get("categories", [])}
    invalid = 0
    for annotation in coco.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        bbox = annotation.get("bbox", [])
        if image is None or annotation.get("category_id") not in categories or len(bbox) != 4:
            invalid += 1
            continue
        x, y, width, height = bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6:
            invalid += 1
    return invalid


def _raster_map(images_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem not in result or path.suffix.lower() == ".pbm":
            result[path.stem] = path
    return result


def prepare_dataset(
    raw_root: Path = RAW_DEFAULT,
    output_root: Path = OUTPUT_DEFAULT,
    overwrite: bool = False,
    report_path: Path | None = None,
) -> PreparationReport:
    """Build PNG/YOLO derivatives and report each data-quality decision."""
    raw_root, output_root = Path(raw_root), Path(output_root)
    report = PreparationReport(str(raw_root), str(output_root))
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_root}")
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Derived dataset already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True)
        (output_root / "labels" / split).mkdir(parents=True)

    records: list[tuple[str, str, Path, Path | None]] = []
    for band in ("SSS_HF_images", "SSS_LF_images"):
        band_root = raw_root / band
        labels_root = band_root / "YOLO_Annotation"
        classes_path = labels_root / "classes.txt"
        if not band_root.exists() or not classes_path.exists() or classes_path.read_text(encoding="utf-8").strip() != EXPECTED_CLASS:
            raise ValueError(f"{band}: expected classes.txt containing only '{EXPECTED_CLASS}'")
        report.coco_invalid_boxes += _validate_coco(band_root / "COCO_Annotation" / "coco_format.json")
        rasters = _raster_map(band_root / "Image")
        label_files = {path.stem: path for path in labels_root.glob("*.txt") if path.name != "classes.txt"}
        report.orphan_label_files += sum(stem not in rasters for stem in label_files)

        for stem, raster in rasters.items():
            records.append((band, stem, raster, label_files.get(stem)))

    # The final 20% of all timestamped frames is held out. A single chronological
    # cutoff is shared by HF and LF data, preventing same-time frames from leaking
    # across the train/validation boundary.
    records.sort(key=lambda record: _timestamp_key(record[1]))
    if not records:
        raise ValueError("No supported sonar raster files were found.")
    cutoff_index = max(1, int(len(records) * 0.8))
    cutoff_key = _timestamp_key(records[cutoff_index - 1][1])
    report.split_cutoff = str(records[cutoff_index - 1][1])
    for band, stem, raster, label_file in records:
        split = "train" if _timestamp_key(stem) <= cutoff_key else "val"
        valid_labels, invalid_lines = _read_valid_labels(label_file) if label_file else ([], 0)
        report.invalid_label_rows += invalid_lines
        try:
            with Image.open(raster) as source:
                image = preprocess_sonar_image(source)
                destination = output_root / "images" / split / f"{band}_{stem}.png"
                image.save(destination, "PNG")
        except (OSError, ValueError):
            report.unreadable_images += 1
            continue
        (output_root / "labels" / split / f"{band}_{stem}.txt").write_text("\n".join(valid_labels) + ("\n" if valid_labels else ""), encoding="utf-8")
        report.images_written += 1
        report.annotation_rows += len(valid_labels)
        report.labelled_images += int(bool(valid_labels))
        if split == "train":
            report.train_images += 1
        else:
            report.val_images += 1

    data_yaml = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: Pipeline\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report_path = report_path or Path("reports") / "preparation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the derived output directory.")
    args = parser.parse_args()
    print(json.dumps(asdict(prepare_dataset(args.raw_root, args.output_root, args.overwrite)), indent=2))


if __name__ == "__main__":
    main()
