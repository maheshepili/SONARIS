"""Run pipeline detection and emit an annotated image plus JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.anomaly.anomaly_score import score_detection
from src.data.preprocess import preprocess_sonar_image
from src.visualization.draw_detections import draw_detections


def result_payload(image_path: Path, detections: list[dict]) -> dict:
    return {"image": str(image_path), "detections": detections, "anomaly_score_type": "HEURISTIC_ANOMALY_SCORE"}


def run_inference(image_path: Path, weights: Path, confidence_threshold: float = 0.25) -> tuple[dict, Image.Image]:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics is not installed") from error
    with Image.open(image_path) as source:
        image = preprocess_sonar_image(source)
    result = YOLO(str(weights)).predict(np.asarray(image), conf=confidence_threshold, verbose=False)[0]
    detections: list[dict] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        bbox = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        anomaly_score, priority, evidence = score_detection(confidence, bbox, image.size)
        detections.append({
            "class": "Pipeline",
            "confidence": round(confidence, 4),
            "bbox": bbox,
            "anomaly_score": anomaly_score,
            "priority": priority,
            "review_label": "Potential anomalous sonar signature — requires analyst review",
            "heuristic_evidence": evidence,
            "reason": "Prototype heuristic: confidence, relative target size, and image-edge context; not marine-debris classification.",
        })
    return result_payload(image_path, detections), draw_detections(image, detections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=Path("reports/training/pipeline_yolo11n/weights/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/predictions"))
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not args.image.exists() or not args.weights.exists():
        raise SystemExit("Image or model weights do not exist. Train the detector before inference.")
    payload, rendered = run_inference(args.image, args.weights, args.confidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    rendered.save(args.output_dir / f"{stem}_annotated.png")
    (args.output_dir / f"{stem}_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
