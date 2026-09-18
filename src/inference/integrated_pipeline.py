"""Combine known Pipeline detection with reconstruction-based anomaly analysis.

YOLO identifies the known Pipeline class.  The autoencoder only identifies
unusual sonar patterns; an autoencoder flag is therefore reported as a
"Potential Anomaly" that requires expert verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.anomaly.extract_candidate_regions import extract_candidate_regions
from src.anomaly.heatmap import save_heatmap
from src.anomaly.score_reconstruction import score_image
from src.inference.prioritization import prioritize_evidence
from src.models.inference import run_inference


DEFAULT_YOLO_WEIGHTS = Path("runs/detect/reports/training/pipeline_yolo11n/weights/best.pt")
DEFAULT_ANOMALY_CHECKPOINT = Path("reports/anomaly/sonar_autoencoder.pt")
DEFAULT_IMAGE_ANOMALY_THRESHOLD = 0.028139928355813026
DEFAULT_PATCH_ANOMALY_THRESHOLD = 0.0038187976460903883
DEFAULT_OUTPUT_DIR = Path("reports/integrated")


def determine_finding(pipeline_detected: bool, anomaly_detected: bool) -> str:
    """Return the combined finding without assigning an anomaly class."""
    if pipeline_detected and anomaly_detected:
        return "Known Pipeline + Potential Anomaly"
    if pipeline_detected:
        return "Known Pipeline"
    if anomaly_detected:
        return "Potential Anomaly"
    return "No Significant Finding"


def run_integrated_pipeline(
    image_path: Path,
    yolo_weights: Path = DEFAULT_YOLO_WEIGHTS,
    anomaly_checkpoint: Path = DEFAULT_ANOMALY_CHECKPOINT,
    image_anomaly_threshold: float = DEFAULT_IMAGE_ANOMALY_THRESHOLD,
    patch_anomaly_threshold: float = DEFAULT_PATCH_ANOMALY_THRESHOLD,
    confidence: float = 0.25,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run the existing YOLO and autoencoder inference components on one image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights do not exist: {yolo_weights}")
    if not anomaly_checkpoint.exists():
        raise FileNotFoundError(f"Anomaly checkpoint does not exist: {anomaly_checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    detection_image = output_dir / f"{image_path.stem}_detections.png"
    heatmap_image = output_dir / f"{image_path.stem}_anomaly_heatmap.png"
    result_path = output_dir / f"{image_path.stem}_integrated_results.json"

    # Both existing components apply preprocess_sonar_image internally.
    detection_payload, rendered_detections = run_inference(image_path, yolo_weights, confidence)
    rendered_detections.save(detection_image)
    reconstruction = score_image(image_path, anomaly_checkpoint, image_anomaly_threshold)
    save_heatmap(image_path, anomaly_checkpoint, patch_anomaly_threshold, heatmap_image)

    detections = detection_payload["detections"]
    pipeline_detected = any(detection["class"] == "Pipeline" for detection in detections)
    pipeline_confidences = [
        detection["confidence"]
        for detection in detections
        if detection["class"] == "Pipeline" and detection.get("confidence") is not None
    ]
    yolo_confidence = max(pipeline_confidences, default=None)
    anomaly_detected = bool(reconstruction["anomalous"])
    candidate_regions = (
        extract_candidate_regions(
            image_path,
            anomaly_checkpoint,
            patch_anomaly_threshold,
        )
        if anomaly_detected
        else []
    )
    prioritization = prioritize_evidence(
        pipeline_detected=pipeline_detected,
        yolo_confidence=yolo_confidence,
        anomaly_detected=anomaly_detected,
        anomaly_excess_over_threshold=reconstruction["excess_over_threshold"],
        image_anomaly_threshold=image_anomaly_threshold,
        candidate_regions=candidate_regions,
    )
    payload = {
        "image": str(image_path),
        "finding": prioritization.finding,
        "priority": prioritization.priority,
        "requires_expert_verification": prioritization.requires_expert_verification,
        "reason": prioritization.reason,
        "evidence": prioritization.evidence,
        "known_object_detection": {
            "detected": pipeline_detected,
            "detections": detections,
        },
        "anomaly_analysis": {
            "detected": anomaly_detected,
            "threshold_type": "IMAGE_LEVEL_MAX_RECONSTRUCTION_ERROR",
            "max_reconstruction_error": reconstruction["max_reconstruction_error"],
            "mean_reconstruction_error": reconstruction["mean_reconstruction_error"],
            "threshold": image_anomaly_threshold,
            "excess_over_threshold": reconstruction["excess_over_threshold"],
            "heatmap_threshold_type": "PATCH_LEVEL_RECONSTRUCTION_ERROR",
            "heatmap_threshold": patch_anomaly_threshold,
            "candidate_regions": candidate_regions,
        },
        "outputs": {
            "detection_image": str(detection_image),
            "anomaly_heatmap": str(heatmap_image),
        },
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--anomaly-checkpoint", type=Path, default=DEFAULT_ANOMALY_CHECKPOINT)
    parser.add_argument(
        "--image-anomaly-threshold",
        type=float,
        default=DEFAULT_IMAGE_ANOMALY_THRESHOLD,
        help="Image-level maximum reconstruction-error threshold for the final finding.",
    )
    parser.add_argument(
        "--patch-anomaly-threshold",
        type=float,
        default=DEFAULT_PATCH_ANOMALY_THRESHOLD,
        help="Patch-level reconstruction-error threshold used to scale the heatmap.",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        result = run_integrated_pipeline(
            image_path=args.image,
            yolo_weights=args.yolo_weights,
            anomaly_checkpoint=args.anomaly_checkpoint,
            image_anomaly_threshold=args.image_anomaly_threshold,
            patch_anomaly_threshold=args.patch_anomaly_threshold,
            confidence=args.confidence,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
