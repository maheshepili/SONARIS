"""Build a leakage-safe Pipeline-only SONARIS and DRISHTI training dataset.

Only the train splits from the two derived source datasets are copied.  The
SONARIS validation split remains in place and is referenced by ``data.yaml`` so
combined-model metrics are comparable with the SONARIS baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SONARIS_DEFAULT = Path("data/processed/sonaris_detection")
DRISHTI_DEFAULT = Path("data/processed/drishti_pipeline_finetune")
OUTPUT_DEFAULT = Path("data/processed/combined_pipeline_detection")
SEED = 42
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pbm", ".pgm", ".png", ".ppm", ".tif", ".tiff"}


def _image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_inputs(sonaris_root: Path, drishti_root: Path) -> None:
    required = (
        sonaris_root / "images" / "train",
        sonaris_root / "labels" / "train",
        sonaris_root / "images" / "val",
        sonaris_root / "labels" / "val",
        drishti_root / "images" / "train",
        drishti_root / "labels" / "train",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Required directory/directories not found: " + ", ".join(missing))


def _unique_output_name(
    source_dataset: str, original_name: str, used_names: set[str], used_stems: set[str]
) -> str:
    """Return a deterministic image name that cannot collide with a YOLO label."""
    original = Path(original_name)
    candidates = [original.name]
    candidates.extend(
        f"{source_dataset}_{index}_{original.name}" for index in range(1, 10_000)
    )
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate not in used_names and candidate_path.stem not in used_stems:
            used_names.add(candidate)
            used_stems.add(candidate_path.stem)
            return candidate
    raise RuntimeError(f"Unable to derive a collision-free name for {original_name}")


def _write_data_yaml(output_root: Path, sonaris_root: Path) -> None:
    validation_images = (sonaris_root / "images" / "val").resolve().as_posix()
    (output_root / "data.yaml").write_text(
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        f"val: {validation_images}\n"
        "names:\n"
        "  0: Pipeline\n",
        encoding="utf-8",
    )


def prepare_combined_pipeline_dataset(
    sonaris_root: Path = SONARIS_DEFAULT,
    drishti_root: Path = DRISHTI_DEFAULT,
    output_root: Path = OUTPUT_DEFAULT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Copy only source train splits into a new dataset and reference SONARIS val."""
    sonaris_root, drishti_root, output_root = map(Path, (sonaris_root, drishti_root, output_root))
    _validate_inputs(sonaris_root, drishti_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output_root}")

    sources = (("sonaris", sonaris_root), ("drishti", drishti_root))
    source_images = {
        source_name: _image_files(source_root / "images" / "train")
        for source_name, source_root in sources
    }
    for source_name, source_root in sources:
        missing_labels = [
            image_path.name
            for image_path in source_images[source_name]
            if not (source_root / "labels" / "train" / f"{image_path.stem}.txt").is_file()
        ]
        if missing_labels:
            raise ValueError(
                f"{source_name} train images missing label files: " + ", ".join(missing_labels)
            )

    output_root.mkdir(parents=True, exist_ok=True)
    images_output = output_root / "images" / "train"
    labels_output = output_root / "labels" / "train"
    images_output.mkdir(parents=True)
    labels_output.mkdir(parents=True)
    used_names: set[str] = set()
    used_stems: set[str] = set()
    records: list[dict[str, str]] = []
    for source_name, source_root in sources:
        for image_path in source_images[source_name]:
            output_name = _unique_output_name(source_name, image_path.name, used_names, used_stems)
            source_label = source_root / "labels" / "train" / f"{image_path.stem}.txt"
            output_label = f"{Path(output_name).stem}.txt"
            shutil.copy2(image_path, images_output / output_name)
            shutil.copy2(source_label, labels_output / output_label)
            records.append(
                {
                    "source_dataset": source_name,
                    "source_split": "train",
                    "source_image": image_path.name,
                    "source_label": source_label.name,
                    "output_image": output_name,
                    "output_label": output_label,
                }
            )

    _write_data_yaml(output_root, sonaris_root)
    manifest = {
        "dataset": "combined_pipeline_detection",
        "classes": {"0": "Pipeline"},
        "seed": seed,
        "rules": {
            "training": "Copied only SONARIS train and leakage-safe DRISHTI train images and labels.",
            "validation": "References the existing SONARIS validation split; DRISHTI validation is not used.",
            "collisions": "Image and YOLO-label stem collisions are renamed deterministically with their source dataset.",
        },
        "source_counts": {source_name: len(images) for source_name, images in source_images.items()},
        "output_counts": {"train_images": len(records), "train_labels": len(records)},
        "validation_reference": {
            "source_dataset": "sonaris",
            "images_dir": str((sonaris_root / "images" / "val")),
            "labels_dir": str((sonaris_root / "labels" / "val")),
            "image_count": len(_image_files(sonaris_root / "images" / "val")),
        },
        "images": records,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonaris-root", type=Path, default=SONARIS_DEFAULT)
    parser.add_argument("--drishti-root", type=Path, default=DRISHTI_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    manifest = prepare_combined_pipeline_dataset(**vars(args))
    print(json.dumps({"output": str(args.output_root), "output_counts": manifest["output_counts"]}, indent=2))


if __name__ == "__main__":
    main()
