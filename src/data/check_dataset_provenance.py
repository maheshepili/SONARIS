"""Read-only provenance and source-frame leakage check for DRISHTI validation.

The only write performed by this module is the requested JSON report.  It never
changes either dataset or the existing prepared SONARIS split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.audit_multiclass_datasets import DRISHTI_SOURCE_FRAME, IMAGE_EXTENSIONS


SONARIS_DEFAULT = Path("data/raw/SubPipeMiniSSS")
DRISHTI_DEFAULT = Path("data/external/drishti/val")
PREPARED_SONARIS_DEFAULT = Path("data/processed/sonaris_detection")
REPORT_DEFAULT = Path("reports/dataset/provenance_report.json")
PREPARED_NAME = re.compile(r"^(SSS_(?:HF|LF)_images)_(\d+\.\d+)$")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_timestamp(image: Path) -> tuple[str, int] | None:
    """Extract the timestamp and crop x-offset from a DRISHTI bg crop name."""
    match = DRISHTI_SOURCE_FRAME.search(image.stem)
    if not match:
        return None
    offset_match = re.search(r"_x(\d+)$", image.stem, re.IGNORECASE)
    return match.group(1), int(offset_match.group(1)) if offset_match else 0


def _raw_sources(sonaris_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((sonaris_root / "DATA").glob("SSS_*_images/Image/*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result[path.stem].append(path)
    return result


def _prepared_splits(prepared_root: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for split in ("train", "val"):
        directory = prepared_root / "images" / split
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            match = PREPARED_NAME.match(path.stem)
            if match:
                result[(match.group(1), match.group(2))] = split
    return result


def _band_from_raw(path: Path) -> str:
    return path.parent.parent.name


def check_provenance(
    sonaris_root: Path = SONARIS_DEFAULT,
    drishti_root: Path = DRISHTI_DEFAULT,
    prepared_sonaris_root: Path = PREPARED_SONARIS_DEFAULT,
) -> dict[str, Any]:
    """Return provenance for every timestamp-referencing DRISHTI validation crop."""
    sonaris_root, drishti_root, prepared_sonaris_root = map(Path, (sonaris_root, drishti_root, prepared_sonaris_root))
    required = (sonaris_root, drishti_root, prepared_sonaris_root)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required path(s) not found: " + ", ".join(missing))

    sources = _raw_sources(sonaris_root)
    splits = _prepared_splits(prepared_sonaris_root)
    drishti_images = sorted(
        path for path in (drishti_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    records: list[dict[str, Any]] = []
    exact_duplicate_pairs: list[dict[str, str]] = []
    source_hashes: dict[Path, str] = {}

    for drishti_image in drishti_images:
        reference = _source_timestamp(drishti_image)
        if reference is None:
            continue
        timestamp, x_offset = reference
        raw_paths = sources.get(timestamp, [])
        drishti_hash = _sha256(drishti_image)
        candidates = []
        for raw_path in raw_paths:
            band = _band_from_raw(raw_path)
            source_hash = source_hashes.setdefault(raw_path, _sha256(raw_path))
            split = splits.get((band, timestamp), "not_in_current_split")
            source_relative = _relative(raw_path, sonaris_root)
            candidates.append({
                "band": band,
                "source_image": source_relative,
                "exists_locally": True,
                "current_split": split,
                "exact_sha256_match": drishti_hash == source_hash,
            })
            if drishti_hash == source_hash:
                exact_duplicate_pairs.append({
                    "drishti_image": _relative(drishti_image, drishti_root),
                    "sonaris_image": source_relative,
                })
        if not candidates:
            status = "UNKNOWN"
            reason = "Filename names a source timestamp, but no local SONARIS raw image has that timestamp."
        elif any(candidate["current_split"] == "train" for candidate in candidates):
            status = "LEAKAGE_RISK"
            reason = "At least one locally available SONARIS source-frame candidate is in the current training split."
        else:
            status = "SAFE_EXTERNAL"
            reason = "No resolved SONARIS source-frame candidate is in the current training split."
        records.append({
            "drishti_image": _relative(drishti_image, drishti_root),
            "provenance": {
                "method": "DRISHTI bg_<SONARIS timestamp>_x<crop offset> filename convention",
                "referenced_sonaris_timestamp": timestamp,
                "crop_x_offset": x_offset,
            },
            "sonaris_source_exists_locally": bool(candidates),
            "sonaris_candidates": candidates,
            "status": status,
            "reason": reason,
        })

    statuses = Counter(record["status"] for record in records)
    return {
        "audit": "drishti_sonaris_provenance",
        "sources": {
            "sonaris_raw": str(sonaris_root),
            "drishti_validation": str(drishti_root),
            "current_sonaris_split": str(prepared_sonaris_root),
        },
        "summary": {
            "drishti_timestamp_referencing_crops": len(records),
            "status_counts": dict(sorted(statuses.items())),
            "exact_sha256_candidate_pairs_checked": sum(len(record["sonaris_candidates"]) for record in records),
            "exact_sha256_duplicate_pair_count": len(exact_duplicate_pairs),
            "exact_sha256_duplicate_pairs": exact_duplicate_pairs,
        },
        "relationships": records,
    }


def _human_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    statuses = summary["status_counts"]
    lines = [
        "DRISHTI to SONARIS provenance check",
        f"Timestamp-referencing DRISHTI crops: {summary['drishti_timestamp_referencing_crops']}",
        f"LEAKAGE_RISK: {statuses.get('LEAKAGE_RISK', 0)} | SAFE_EXTERNAL: {statuses.get('SAFE_EXTERNAL', 0)} | UNKNOWN: {statuses.get('UNKNOWN', 0)}",
        f"Exact SHA-256 source candidates checked: {summary['exact_sha256_candidate_pairs_checked']}; exact duplicate pairs: {summary['exact_sha256_duplicate_pair_count']}",
    ]
    if statuses.get("LEAKAGE_RISK", 0):
        lines.append("Conclusion: DRISHTI validation is not independent of the current SONARIS training split and must not be used as an external validation set unchanged.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonaris-root", type=Path, default=SONARIS_DEFAULT)
    parser.add_argument("--drishti-root", type=Path, default=DRISHTI_DEFAULT)
    parser.add_argument("--prepared-sonaris-root", type=Path, default=PREPARED_SONARIS_DEFAULT)
    parser.add_argument("--report-path", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    report = check_provenance(args.sonaris_root, args.drishti_root, args.prepared_sonaris_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(_human_summary(report))
    print(f"JSON report: {args.report_path}")


if __name__ == "__main__":
    main()
