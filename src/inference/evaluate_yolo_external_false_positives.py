from pathlib import Path
import argparse
from collections import defaultdict
import json

from src.models.inference import run_inference


CLASS_NAMES = {
    0: "Pipeline",
    1: "Shipwreck",
    2: "Ghost Net",
    3: "Mine Cylinder",
}


def read_ground_truth(label_path):
    classes = set()

    if not label_path.exists():
        return classes

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if not parts:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if class_id in CLASS_NAMES:
            classes.add(class_id)

    return classes


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pipeline-only YOLO detections against external ground-truth classes."
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/processed/sonaris_multiclass_clean_v2/images/external_validation"
        ),
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "data/processed/sonaris_multiclass_clean_v2/labels/external_validation"
        ),
    )

    parser.add_argument(
        "--yolo-weights",
        default="runs/detect/reports/training/pipeline_yolo11n/weights/best.pt",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    image_files = sorted(
        p
        for p in args.images.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    stats = defaultdict(
        lambda: {
            "images": 0,
            "pipeline_detected": 0,
        }
    )

    evaluated = 0
    skipped_no_label = 0

    for index, image_path in enumerate(image_files, start=1):

        label_path = args.labels / f"{image_path.stem}.txt"
        gt_classes = read_ground_truth(label_path)

        if not gt_classes:
            skipped_no_label += 1
            continue

        payload, _ = run_inference(
            image_path,
            weights=args.yolo_weights,
            confidence_threshold=args.confidence,
        )

        detections = payload["detections"]

        pipeline_detected = any(
            detection.get("class") == "Pipeline"
            for detection in detections
        )

        for class_id in gt_classes:
            stats[class_id]["images"] += 1

            if pipeline_detected:
                stats[class_id]["pipeline_detected"] += 1

        evaluated += 1

        if index % 50 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)}")

    print("\nYOLO EXTERNAL FALSE-POSITIVE ANALYSIS")
    print("=" * 55)
    print(f"Images discovered: {len(image_files)}")
    print(f"Images evaluated:  {evaluated}")
    print(f"Skipped/no labels: {skipped_no_label}")
    print(f"YOLO confidence:   {args.confidence}")

    results = {
        "images_discovered": len(image_files),
        "images_evaluated": evaluated,
        "skipped_no_label": skipped_no_label,
        "confidence": args.confidence,
        "classes": {},
    }

    for class_id in sorted(CLASS_NAMES):

        total = stats[class_id]["images"]
        detected = stats[class_id]["pipeline_detected"]

        rate = detected / total if total else 0.0

        results["classes"][CLASS_NAMES[class_id]] = {
            "ground_truth_images": total,
            "pipeline_detected_images": detected,
            "pipeline_detection_rate": round(rate, 6),
            "no_pipeline_detection_images": total - detected,
        }

        print(
            f"{CLASS_NAMES[class_id]:15s}: "
            f"{detected}/{total} Pipeline detections "
            f"({rate * 100:.2f}%)"
        )

    if args.output:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()