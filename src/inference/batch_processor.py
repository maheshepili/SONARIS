"""Run the integrated SONARIS inference pipeline over an image directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.inference.integrated_pipeline import (
    DEFAULT_OUTPUT_DIR,
    run_integrated_pipeline,
)


SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
DEFAULT_BATCH_RESULT_NAME = "batch_results.json"


def iter_supported_images(input_dir: Path) -> Iterator[Path]:
    """Yield supported image files in ``input_dir`` without opening them."""
    for image_path in input_dir.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ):
            yield image_path


def _batch_record(pipeline_result: dict[str, Any], filename: str) -> dict[str, Any]:
    """Keep the per-image fields relevant to a batch report."""
    anomaly_result = pipeline_result["anomaly_analysis"]
    record: dict[str, Any] = {
        "filename": filename,
        "finding": pipeline_result["finding"],
        "priority": pipeline_result["priority"],
        "reason": pipeline_result["reason"],
        "requires_expert_verification": pipeline_result["requires_expert_verification"],
        "evidence": pipeline_result["evidence"],
        "known_object_detection": pipeline_result["known_object_detection"],
        "anomaly_result": anomaly_result,
    }

    # Preserve candidate regions at the batch-record level for consumers that
    # read localization evidence without traversing anomaly_result.
    candidate_regions = pipeline_result.get(
        "candidate_regions", anomaly_result.get("candidate_regions")
    )
    if candidate_regions is not None:
        record["candidate_regions"] = candidate_regions
    return record


def process_batch(input_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Process supported images sequentially and save a batch result report."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    records: list[dict[str, Any]] = []
    finding_counts: Counter[str] = Counter()

    for image_path in iter_supported_images(input_dir):
        pipeline_result = run_integrated_pipeline(
            image_path=image_path,
            output_dir=output_dir,
        )
        record = _batch_record(pipeline_result, image_path.name)
        records.append(record)
        finding_counts[record["finding"]] += 1

    summary = {
        "total_images": len(records),
        "finding_counts": dict(finding_counts),
    }
    batch_result = {"summary": summary, "images": records}

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / DEFAULT_BATCH_RESULT_NAME
    result_path.write_text(json.dumps(batch_result, indent=2), encoding="utf-8")
    return batch_result


def main() -> None:
    """Run batch inference from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        result = process_batch(args.input_dir, args.output_dir)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved batch results: {args.output_dir / DEFAULT_BATCH_RESULT_NAME}")


if __name__ == "__main__":
    main()
