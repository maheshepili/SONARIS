"""Render annotated FLM-MLDT samples without modifying the source dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from src.data.flm_mldt import CLASSES, discover, _parse, DEFAULT_IMAGES, DEFAULT_ANNOTATIONS

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=Path("reports/flm_mldt_samples"))
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records, _, _ = discover(args.images_root, args.annotations_root)
    # Spread deterministic samples through the corpus instead of selecting adjacent frames.
    picks = [records[i * (len(records) - 1) // max(1, args.count - 1)] for i in range(min(args.count, len(records)))]
    for number, record in enumerate(picks):
        image = Image.open(record.image).convert("RGB"); draw = ImageDraw.Draw(image)
        for class_id, x, y, width, height in _parse(record.label)[0]:
            left, top = (x - width / 2) * image.width, (y - height / 2) * image.height
            right, bottom = (x + width / 2) * image.width, (y + height / 2) * image.height
            text = CLASSES[class_id] if class_id < len(CLASSES) else f"UNKNOWN ID {class_id}"
            draw.rectangle((left, top, right, bottom), outline="red", width=3); draw.text((left, max(0, top - 14)), text, fill="red")
        image.save(args.output / f"{number:02d}_{record.sequence.replace('/', '_')}_{record.image.name}")

if __name__ == "__main__": main()
