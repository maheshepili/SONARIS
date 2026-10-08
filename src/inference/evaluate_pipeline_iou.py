from pathlib import Path
import argparse
from collections import defaultdict
import json
from PIL import Image

from src.models.inference import run_inference


def read_pipeline_boxes(label_path, image_path):
    boxes = []

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        if class_id != 0:
            continue

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes.append([x1, y1, x2, y2])

    return boxes

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pipeline YOLO localization against external Pipeline labels."
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

    total_gt = 0
    total_predictions = 0
    matched_predictions = 0

    best_ious = []

    pipeline_images = 0
    pipeline_images_detected = 0

    skipped = 0

    for index, image_path in enumerate(image_files, start=1):

        label_path = args.labels / f"{image_path.stem}.txt"
        gt_boxes = read_pipeline_boxes(label_path, image_path)

        # Only evaluate images containing actual Pipeline ground truth.
        if not gt_boxes:
            continue

        pipeline_images += 1
        total_gt += len(gt_boxes)

        payload, _ = run_inference(
            image_path,
            weights=args.yolo_weights,
            confidence_threshold=args.confidence,
        )

        predictions = [
            detection
            for detection in payload["detections"]
            if detection.get("class") == "Pipeline"
        ]

        total_predictions += len(predictions)

        prediction_boxes = []

        for detection in predictions:
            x, y, width, height = detection["bbox"]

            prediction_boxes.append(
                [
                    x,
                    y,
                    x + width,
                    y + height,
                ]
            )

        image_best_iou = 0.0

        for gt_box in gt_boxes:
            best_iou = 0.0

            for prediction_box in prediction_boxes:
                best_iou = max(
                    best_iou,
                    iou(gt_box, prediction_box),
                )

            image_best_iou = max(image_best_iou, best_iou)
            best_ious.append(best_iou)

            if best_iou >= 0.5:
                matched_predictions += 1

        if predictions:
            pipeline_images_detected += 1

        if index % 50 == 0 or index == len(image_files):
            print(f"Processed {index}/{len(image_files)}")

    mean_iou = (
        sum(best_ious) / len(best_ious)
        if best_ious
        else 0.0
    )

    results = {
        "confidence": args.confidence,
        "pipeline_images": pipeline_images,
        "pipeline_ground_truth_boxes": total_gt,
        "pipeline_predictions": total_predictions,
        "pipeline_images_with_detection": pipeline_images_detected,
        "image_level_detection_rate": (
            pipeline_images_detected / pipeline_images
            if pipeline_images
            else 0.0
        ),
        "ground_truth_boxes_iou_ge_0.5": matched_predictions,
        "box_recall_iou_ge_0.5": (
            matched_predictions / total_gt
            if total_gt
            else 0.0
        ),
        "mean_best_iou": mean_iou,
        "evaluated_boxes": len(best_ious),
    }

    print("\nPIPELINE YOLO IoU EVALUATION")
    print("=" * 55)
    print(f"Pipeline images:             {pipeline_images}")
    print(f"Ground-truth Pipeline boxes: {total_gt}")
    print(f"YOLO predictions:             {total_predictions}")
    print(f"Images with detection:        {pipeline_images_detected}")
    print(
        f"Image detection rate:         "
        f"{results['image_level_detection_rate'] * 100:.2f}%"
    )
    print(
        f"IoU >= 0.50:                  "
        f"{matched_predictions}/{total_gt} "
        f"({results['box_recall_iou_ge_0.5'] * 100:.2f}%)"
    )
    print(f"Mean best IoU:                {mean_iou:.4f}")

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