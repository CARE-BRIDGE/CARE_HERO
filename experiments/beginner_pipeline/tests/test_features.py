from __future__ import annotations

from features import (
    FrameObservation,
    Landmark,
    calculate_front_facing,
    classify_coverage,
    summarize_observations,
)


def _point(x: float, y: float, visibility: float = 0.95) -> Landmark:
    return Landmark(x=x, y=y, visibility=visibility)


def test_upper_body_is_not_mistaken_for_full_body() -> None:
    landmarks = {
        "nose": _point(0.50, 0.20),
        "left_shoulder": _point(0.40, 0.35),
        "right_shoulder": _point(0.60, 0.35),
        "left_knee": _point(0.45, 0.90, 0.10),
        "right_knee": _point(0.55, 0.90, 0.10),
    }
    assert classify_coverage(landmarks) == "upper_body"


def test_front_facing_uses_nose_and_shoulder_center() -> None:
    landmarks = {
        "nose": _point(0.50, 0.20),
        "left_shoulder": _point(0.40, 0.35),
        "right_shoulder": _point(0.60, 0.35),
    }
    assert calculate_front_facing(landmarks) == 1.0


def test_summary_keeps_unavailable_torso_value_as_none() -> None:
    observation = FrameObservation(
        pose_detected=True,
        face_landmark_count=478,
        landmarks={},
        coverage="upper_body",
        front_facing=1.0,
        torso_tilt_degrees=None,
    )
    summary = summarize_observations([observation, observation])
    assert summary["analysis_scope"] == "upper_body"
    assert summary["features"]["mean_abs_torso_tilt_deg"] is None

