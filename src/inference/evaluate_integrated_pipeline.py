from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from src.anomaly.score_reconstruction import load_model, score_image
from src.models.inference import run_inference


DEFAULT_IMAGE_ANOMALY_THRESHOLD = 0.028139928355813026
DEFAULT_ANOMALY_CHECKPOINT = Path(
    "reports/anomaly/sonar_autoencoder.pt"
)
DEFAULT_YOLO_WEIGHTS = Path(
    "runs/detect/reports/training/pipeline_yolo11n/weights/best.pt"
)


def classify_finding(
    known_pipeline: bool,
    anomaly: bool,
) -> str:
    if known_pipeline and anomaly:
        return "Known Pipeline + Potential Anomaly"
    if known_pipeline:
        return "Known Pipeline"
    if anomaly:
        return "Potential Anomaly"
    return "No Significant Finding"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate integrated YOLO + autoencoder pipeline."
    )

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
    )

    parser.add_argument(
        "--anomaly-checkpoint",
        type=Path,
        default=DEFAULT_ANOMALY_CHECKPOINT,
    )

    parser.add_argument(
        "--image-anomaly-threshold",
        type=float,
        default=DEFAULT_IMAGE_ANOMALY_THRESHOLD,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/integrated/"
            "integrated_validation_summary.json"
        ),
    )

    args = parser.parse_args()

    if not args.images.exists():
        raise SystemExit(
            f"Validation image directory does not exist: "
            f"{args.images}"
        )

    if not args.yolo_weights.exists():
        raise SystemExit(
            f"YOLO weights do not exist: "
            f"{args.yolo_weights}"
        )

    if not args.anomaly_checkpoint.exists():
        raise SystemExit(
            f"Anomaly checkpoint does not exist: "
            f"{args.anomaly_checkpoint}"
        )

    images = sorted(args.images.glob("*.png"))

    if not images:
        raise SystemExit(
            f"No PNG images found in {args.images}"
        )

    print(f"Validation images: {len(images)}")
    print("Loading autoencoder once...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    anomaly_model, patch_size = load_model(
        args.anomaly_checkpoint,
        device,
    )

    findings = Counter()
    anomaly_count = 0
    pipeline_count = 0

    max_errors = []
    mean_errors = []

    for index, image_path in enumerate(images, start=1):
        print(
            f"\rProcessing {index}/{len(images)} "
            f"{image_path.name}",
            end="",
            flush=True,
        )

        detection_result, _ = run_inference(
            image_path=image_path,
            weights=args.yolo_weights,
            confidence_threshold=args.confidence,
        )

        detections = detection_result.get("detections", [])
        known_pipeline = any(
            detection["class"] == "Pipeline"
            for detection in detections
        )

        anomaly = score_image(
            image_path=image_path,
            checkpoint=args.anomaly_checkpoint,
            threshold=args.image_anomaly_threshold,
            model=anomaly_model,
            patch_size=patch_size,
            device=device,
        )

        anomaly_detected = bool(
            anomaly["anomalous"]
        )

        finding = classify_finding(
            known_pipeline,
            anomaly_detected,
        )

        findings[finding] += 1

        if known_pipeline:
            pipeline_count += 1

        if anomaly_detected:
            anomaly_count += 1

        max_errors.append(
            float(
                anomaly["max_reconstruction_error"]
            )
        )

        mean_errors.append(
            float(
                anomaly["mean_reconstruction_error"]
            )
        )

    print("\n")

    total = len(images)

    summary = {
        "images_evaluated": total,
        "threshold_type":
            "IMAGE_LEVEL_MAX_RECONSTRUCTION_ERROR",
        "image_anomaly_threshold":
            args.image_anomaly_threshold,
        "findings": dict(findings),
        "known_pipeline_images":
            pipeline_count,
        "anomaly_flagged_images":
            anomaly_count,
        "anomaly_flagged_fraction":
            round(anomaly_count / total, 6),
        "max_reconstruction_error_mean":
            round(
                sum(max_errors) / len(max_errors),
                8,
            ),
        "mean_reconstruction_error_mean":
            round(
                sum(mean_errors) / len(mean_errors),
                8,
            ),
        "models": {
            "yolo_weights":
                str(args.yolo_weights),
            "anomaly_checkpoint":
                str(args.anomaly_checkpoint),
            "device":
                str(device),
            "patch_size":
                patch_size,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(
        f"\nSaved summary: {args.output}"
    )


if __name__ == "__main__":
    main()
