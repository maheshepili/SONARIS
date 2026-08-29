"""Explainable, uncalibrated suspicion scoring for detected pipelines."""

from __future__ import annotations


def score_detection(confidence: float, bbox: list[float], image_size: tuple[int, int]) -> tuple[float, str, dict[str, float]]:
    """Score uncertainty, relative size and edge context for analyst triage.

    This is deliberately a heuristic score, not a trained anomaly detector.
    """
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if len(bbox) != 4 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("bbox must contain four values and image size must be positive")
    x, y, width, height = bbox
    image_width, image_height = image_size
    area_ratio = max(0.0, width * height) / (image_width * image_height)
    relative_size = min(area_ratio / 0.10, 1.0)
    margin_x = min(x, image_width - (x + width))
    margin_y = min(y, image_height - (y + height))
    edge_context = 1.0 if min(margin_x / image_width, margin_y / image_height) < 0.03 else 0.0
    score = round(min(1.0, 0.60 * (1.0 - confidence) + 0.25 * relative_size + 0.15 * edge_context), 3)
    priority = "HIGH" if score >= 0.65 else "MEDIUM" if score >= 0.35 else "LOW"
    evidence = {"low_confidence": round(1.0 - confidence, 3), "relative_size": round(relative_size, 3), "edge_context": edge_context}
    return score, priority, evidence
