import json
from pathlib import Path


DATASET_PATH = Path("data/raw/SubPipeMiniSSS/DATA")


def inspect_invalid_annotations(json_path):
    print("\n" + "=" * 70)
    print(f"INVALID ANNOTATIONS: {json_path}")
    print("=" * 70)

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = {image["id"]: image for image in data["images"]}

    invalid_count = 0

    for annotation in data["annotations"]:
        image = images.get(annotation["image_id"])

        if image is None:
            continue

        x, y, width, height = annotation["bbox"]

        invalid = (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image["width"] + 1e-6
            or y + height > image["height"] + 1e-6
        )

        if invalid:
            invalid_count += 1

            print(f"\nAnnotation ID: {annotation['id']}")
            print(f"Image ID:      {annotation['image_id']}")
            print(f"Image file:    {image['file_name']}")
            print(f"Image size:    {image['width']} x {image['height']}")
            print(f"BBox:          {annotation['bbox']}")
            print(
                f"Right edge:    {x + width:.2f} "
                f"/ image width {image['width']}"
            )
            print(
                f"Bottom edge:   {y + height:.2f} "
                f"/ image height {image['height']}"
            )

    print(f"\nTotal invalid annotations: {invalid_count}")


hf_json = (
    DATASET_PATH
    / "SSS_HF_images"
    / "COCO_Annotation"
    / "coco_format.json"
)

lf_json = (
    DATASET_PATH
    / "SSS_LF_images"
    / "COCO_Annotation"
    / "coco_format.json"
)

inspect_invalid_annotations(hf_json)
inspect_invalid_annotations(lf_json)