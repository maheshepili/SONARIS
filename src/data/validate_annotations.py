import json
from pathlib import Path


DATASET_PATH = Path("data/raw/SubPipeMiniSSS/DATA")

TOLERANCE = 1e-6


def validate_annotation_file(json_path):
    print("\n" + "=" * 60)
    print(f"VALIDATING: {json_path}")
    print("=" * 60)

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    image_lookup = {image["id"]: image for image in images}
    annotated_image_ids = set()

    invalid_boxes = 0
    missing_images = 0

    for annotation in annotations:

        image_id = annotation["image_id"]
        annotated_image_ids.add(image_id)

        image = image_lookup.get(image_id)

        if image is None:
            missing_images += 1
            continue

        x, y, width, height = annotation["bbox"]

        invalid = (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image["width"] + TOLERANCE
            or y + height > image["height"] + TOLERANCE
        )

        if invalid:
            invalid_boxes += 1

    empty_images = len(images) - len(annotated_image_ids)

    print(f"Images:                 {len(images)}")
    print(f"Annotations:            {len(annotations)}")
    print(f"Categories:             {len(categories)}")
    print(f"Annotated images:       {len(annotated_image_ids)}")
    print(f"Images without boxes:   {empty_images}")
    print(f"Missing image IDs:      {missing_images}")
    print(f"Invalid bounding boxes: {invalid_boxes}")

    print("\nCategories:")

    for category in categories:
        print(f"  {category['id']}: {category['name']}")


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


validate_annotation_file(hf_json)
validate_annotation_file(lf_json)