"""Body-coverage classification from normalized pose landmark visibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


CORE_UPPER_BODY = ("nose", "left_shoulder", "right_shoulder")
LOWER_BODY = ("left_knee", "right_knee", "left_ankle", "right_ankle")


@dataclass(frozen=True)
class BodyCoverageSummary:
    """Aggregate camera framing without inferring hidden body positions."""

    profile: str
    detected_frame_count: int
    full_body_frame_count: int
    upper_body_frame_count: int
    mixed_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_frame_coverage(
    *,
    detected: bool,
    landmarks: Mapping[str, Any],
    minimum_visibility: float = 0.5,
) -> str:
    """Classify visible framing using only observed landmark visibility."""

    if not detected:
        return "detection_failed"
    upper_visible = all(
        _visible(landmarks.get(name), minimum_visibility)
        for name in CORE_UPPER_BODY
    )
    if not upper_visible:
        return "mixed"
    lower_visible_count = sum(
        _visible(landmarks.get(name), minimum_visibility)
        for name in LOWER_BODY
    )
    if lower_visible_count >= 3:
        return "full_body"
    if lower_visible_count <= 1:
        return "upper_body"
    return "mixed"


def summarize_body_coverage(
    frames: Iterable[Any],
    minimum_visibility: float = 0.5,
    majority_ratio: float = 0.6,
) -> BodyCoverageSummary:
    """Summarize frame-level coverage with an explicit mixed state."""

    counts = {"full_body": 0, "upper_body": 0, "mixed": 0}
    detected_count = 0
    for frame in frames:
        coverage = classify_frame_coverage(
            detected=bool(frame.detected),
            landmarks=frame.landmarks,
            minimum_visibility=minimum_visibility,
        )
        if coverage == "detection_failed":
            continue
        detected_count += 1
        counts[coverage] += 1

    if detected_count == 0:
        profile = "detection_failed"
    elif counts["full_body"] / detected_count >= majority_ratio:
        profile = "full_body"
    elif counts["upper_body"] / detected_count >= majority_ratio:
        profile = "upper_body"
    else:
        profile = "mixed"
    return BodyCoverageSummary(
        profile=profile,
        detected_frame_count=detected_count,
        full_body_frame_count=counts["full_body"],
        upper_body_frame_count=counts["upper_body"],
        mixed_frame_count=counts["mixed"],
    )


def _visible(point: Any, minimum_visibility: float) -> bool:
    return point is not None and float(point.visibility) >= minimum_visibility
