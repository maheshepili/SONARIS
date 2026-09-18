"""Deterministic analyst-review prioritization for SONARIS evidence.

This module consumes evidence already produced by SONARIS; it does not run or
alter detection, reconstruction, preprocessing, thresholds, or localization.
Its output is a transparent review priority, not an ML prediction or a
probability.  It does not determine whether an image presents an actual
danger or threat.

Rules
-----
* No Pipeline detection and no anomaly flag: ``No Significant Finding`` / LOW.
* Pipeline-only: ``Known Pipeline`` / LOW for a sufficiently confident
  detection, otherwise MEDIUM.  These findings do not require expert anomaly
  verification.
* Anomaly-only: ``Potential Anomaly`` / HIGH when its excess is large relative
  to the configured image-level threshold; otherwise MEDIUM.  Expert
  verification is always required.
* Pipeline plus anomaly: ``Known Pipeline + Potential Anomaly`` / HIGH and
  requires expert verification.

Candidate regions are preserved as supporting evidence when available.  They
do not independently change priority: a candidate region is localization
support, while the image-level anomaly flag and its threshold excess determine
whether anomaly evidence is present and strong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence


Priority = Literal["LOW", "MEDIUM", "HIGH"]

# These conservative, explicit policy cut-offs are intentionally separate from
# inference thresholds and may be adjusted without changing the ML pipeline.
LOW_PRIORITY_PIPELINE_CONFIDENCE = 0.85
"""YOLO confidence at or above which a Pipeline-only finding is LOW."""

HIGH_PRIORITY_ANOMALY_EXCESS_RATIO = 0.50
"""Excess/threshold ratio needed for HIGH priority in anomaly-only findings."""


@dataclass(frozen=True)
class PrioritizationResult:
    """A deterministic analyst-review result derived from supplied evidence."""

    finding: str
    priority: Priority
    requires_expert_verification: bool
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this result."""
        return asdict(self)


def prioritize_evidence(
    *,
    pipeline_detected: bool,
    yolo_confidence: float | None,
    anomaly_detected: bool,
    anomaly_excess_over_threshold: float | None,
    image_anomaly_threshold: float | None,
    candidate_regions: Sequence[Mapping[str, Any]] | None = None,
) -> PrioritizationResult:
    """Prioritize an image for analyst review using existing pipeline evidence.

    ``anomaly_excess_over_threshold`` and ``image_anomaly_threshold`` are used
    only when both are supplied.  An anomaly flag with unavailable numeric
    excess is retained as a MEDIUM, expert-review finding rather than being
    promoted to HIGH.  This makes incomplete evidence conservative.

    Raises:
        ValueError: If supplied confidence, excess, or threshold is outside
            its valid range.
    """
    _validate_confidence(yolo_confidence)
    _validate_anomaly_values(
        anomaly_excess_over_threshold, image_anomaly_threshold
    )

    evidence = _build_evidence(
        pipeline_detected=pipeline_detected,
        yolo_confidence=yolo_confidence,
        anomaly_detected=anomaly_detected,
        anomaly_excess_over_threshold=anomaly_excess_over_threshold,
        image_anomaly_threshold=image_anomaly_threshold,
        candidate_regions=candidate_regions,
    )

    if pipeline_detected and anomaly_detected:
        return PrioritizationResult(
            finding="Known Pipeline + Potential Anomaly",
            priority="HIGH",
            requires_expert_verification=True,
            reason=(
                "A known Pipeline detection and image-level anomaly evidence "
                "are both present; prioritize expert verification."
            ),
            evidence=evidence,
        )

    if anomaly_detected:
        if _has_strong_anomaly_excess(
            anomaly_excess_over_threshold, image_anomaly_threshold
        ):
            return PrioritizationResult(
                finding="Potential Anomaly",
                priority="HIGH",
                requires_expert_verification=True,
                reason=(
                    "Image-level anomaly evidence exceeds the configured "
                    "threshold by the high-priority policy ratio; expert "
                    "verification is required."
                ),
                evidence=evidence,
            )
        return PrioritizationResult(
            finding="Potential Anomaly",
            priority="MEDIUM",
            requires_expert_verification=True,
            reason=(
                "Image-level anomaly evidence is present but does not meet "
                "the high-priority excess policy; expert verification is "
                "required."
            ),
            evidence=evidence,
        )

    if pipeline_detected:
        if (
            yolo_confidence is not None
            and yolo_confidence >= LOW_PRIORITY_PIPELINE_CONFIDENCE
        ):
            return PrioritizationResult(
                finding="Known Pipeline",
                priority="LOW",
                requires_expert_verification=False,
                reason=(
                    "Only a sufficiently confident known Pipeline detection "
                    "is present; no image-level anomaly evidence was supplied."
                ),
                evidence=evidence,
            )
        return PrioritizationResult(
            finding="Known Pipeline",
            priority="MEDIUM",
            requires_expert_verification=False,
            reason=(
                "Only a known Pipeline detection is present, but its confidence "
                "is below the low-priority policy cut-off or unavailable."
            ),
            evidence=evidence,
        )

    return PrioritizationResult(
        finding="No Significant Finding",
        priority="LOW",
        requires_expert_verification=False,
        reason=(
            "No known Pipeline detection or image-level anomaly evidence is "
            "present."
        ),
        evidence=evidence,
    )


def _has_strong_anomaly_excess(
    excess: float | None, threshold: float | None
) -> bool:
    """Return whether available anomaly evidence meets the HIGH policy cut-off."""
    return (
        excess is not None
        and threshold is not None
        and excess / threshold >= HIGH_PRIORITY_ANOMALY_EXCESS_RATIO
    )


def _build_evidence(
    *,
    pipeline_detected: bool,
    yolo_confidence: float | None,
    anomaly_detected: bool,
    anomaly_excess_over_threshold: float | None,
    image_anomaly_threshold: float | None,
    candidate_regions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Build the complete, inspectable evidence record without inference."""
    evidence: dict[str, Any] = {
        "pipeline_detected": pipeline_detected,
        "yolo_confidence": yolo_confidence,
        "anomaly_detected": anomaly_detected,
        "anomaly_excess_over_threshold": anomaly_excess_over_threshold,
        "image_anomaly_threshold": image_anomaly_threshold,
    }
    if anomaly_excess_over_threshold is not None and image_anomaly_threshold is not None:
        evidence["anomaly_excess_ratio"] = (
            anomaly_excess_over_threshold / image_anomaly_threshold
        )
    if candidate_regions is not None:
        # Copy to make the returned evidence independent of the caller's list.
        evidence["candidate_regions"] = [dict(region) for region in candidate_regions]
        evidence["candidate_region_count"] = len(candidate_regions)
    return evidence


def _validate_confidence(confidence: float | None) -> None:
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError("yolo_confidence must be between 0 and 1")


def _validate_anomaly_values(excess: float | None, threshold: float | None) -> None:
    if excess is not None and excess < 0.0:
        raise ValueError("anomaly_excess_over_threshold must be non-negative")
    if threshold is not None and threshold <= 0.0:
        raise ValueError("image_anomaly_threshold must be greater than zero")
