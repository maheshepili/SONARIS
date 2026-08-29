"""Render detection results with Pillow."""

from __future__ import annotations

from PIL import Image, ImageDraw


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(canvas)
    for detection in detections:
        x, y, width, height = detection["bbox"]
        priority = detection["priority"]
        colour = {"HIGH": "red", "MEDIUM": "orange", "LOW": "lime"}[priority]
        drawer.rectangle((x, y, x + width, y + height), outline=colour, width=3)
        drawer.text((x, max(0, y - 14)), f"Pipeline {detection['confidence']:.2f} | {priority}", fill=colour)
    return canvas
