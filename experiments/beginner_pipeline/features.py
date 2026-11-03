"""MediaPipe 좌표를 사람이 설명할 수 있는 숫자 특징으로 바꾸는 함수 모음."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class FrameObservation:
    pose_detected: bool
    face_landmark_count: int
    landmarks: Mapping[str, Landmark]
    coverage: str
    front_facing: float | None
    torso_tilt_degrees: float | None


def classify_coverage(
    landmarks: Mapping[str, Landmark], minimum_visibility: float = 0.5
) -> str:
    """검출된 좌표만 보고 전신/상반신/혼합 촬영을 구분한다."""

    upper_names = ("nose", "left_shoulder", "right_shoulder")
    if not all(_visible(landmarks.get(name), minimum_visibility) for name in upper_names):
        return "mixed"

    lower_names = ("left_knee", "right_knee", "left_ankle", "right_ankle")
    visible_lower_count = sum(
        _visible(landmarks.get(name), minimum_visibility) for name in lower_names
    )
    if visible_lower_count >= 3:
        return "full_body"
    if visible_lower_count <= 1:
        return "upper_body"
    return "mixed"


def calculate_front_facing(
    landmarks: Mapping[str, Landmark],
    *,
    minimum_visibility: float = 0.5,
    maximum_offset: float = 0.35,
) -> float | None:
    """코가 양 어깨 중앙에서 크게 벗어나지 않았는지 0 또는 1로 표현한다."""

    points = _required_points(
        landmarks,
        ("nose", "left_shoulder", "right_shoulder"),
        minimum_visibility,
    )
    if points is None:
        return None
    nose, left_shoulder, right_shoulder = points
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
    shoulder_width = math.hypot(
        right_shoulder.x - left_shoulder.x,
        right_shoulder.y - left_shoulder.y,
    )
    if shoulder_width <= 1e-9:
        return None
    normalized_offset = abs(nose.x - shoulder_center_x) / shoulder_width
    return 1.0 if normalized_offset <= maximum_offset else 0.0


def calculate_torso_tilt(
    landmarks: Mapping[str, Landmark], minimum_visibility: float = 0.5
) -> float | None:
    """어깨 중앙과 골반 중앙을 이은 선의 좌우 기울기를 각도로 계산한다."""

    points = _required_points(
        landmarks,
        ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        minimum_visibility,
    )
    if points is None:
        return None
    left_shoulder, right_shoulder, left_hip, right_hip = points
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_x = (left_hip.x + right_hip.x) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0
    dx = shoulder_x - hip_x
    dy_up = hip_y - shoulder_y
    if math.hypot(dx, dy_up) <= 1e-9:
        return None
    return math.degrees(math.atan2(dx, dy_up))


def summarize_observations(observations: list[FrameObservation]) -> dict[str, object]:
    """여러 프레임의 관찰값을 모델 입력용 한 행으로 요약한다."""

    total = len(observations)
    if total == 0:
        raise ValueError("분석된 프레임이 없습니다.")

    pose_count = sum(item.pose_detected for item in observations)
    face_count = sum(item.face_landmark_count > 0 for item in observations)
    front_values = [
        item.front_facing for item in observations if item.front_facing is not None
    ]
    tilt_values = [
        item.torso_tilt_degrees
        for item in observations
        if item.torso_tilt_degrees is not None
    ]
    upper_body_count = sum(item.coverage == "upper_body" for item in observations)
    full_body_count = sum(item.coverage == "full_body" for item in observations)
    mixed_count = sum(item.coverage == "mixed" for item in observations)

    if upper_body_count / total >= 0.6:
        analysis_scope = "upper_body"
    elif full_body_count / total >= 0.6:
        analysis_scope = "full_body"
    else:
        analysis_scope = "mixed"

    feature_values = {
        "pose_detection_rate": pose_count / total,
        "face_detection_rate": face_count / total,
        "front_facing_ratio": _mean_or_none(front_values),
        "mean_abs_torso_tilt_deg": _mean_or_none([abs(value) for value in tilt_values]),
        "torso_tilt_std_deg": (
            float(statistics.pstdev(tilt_values)) if len(tilt_values) >= 2 else None
        ),
        "upper_body_ratio": upper_body_count / total,
    }
    return {
        "analyzed_frame_count": total,
        "pose_detected_frame_count": pose_count,
        "face_detected_frame_count": face_count,
        "coverage_counts": {
            "full_body": full_body_count,
            "upper_body": upper_body_count,
            "mixed": mixed_count,
        },
        "analysis_scope": analysis_scope,
        "features": feature_values,
        "limitations": _limitations(analysis_scope, feature_values),
    }


def _limitations(scope: str, features: Mapping[str, object]) -> list[str]:
    messages = [
        "2D 카메라 좌표 기반 결과이며 실제 거리·시선·감정·의도를 직접 측정하지 않습니다."
    ]
    if scope == "upper_body":
        messages.append("상반신 촬영이므로 하반신 지표는 평가하지 않습니다.")
    if features["mean_abs_torso_tilt_deg"] is None:
        messages.append("어깨와 골반 좌표가 충분하지 않아 몸통 기울기를 계산하지 않았습니다.")
    return messages


def _required_points(
    landmarks: Mapping[str, Landmark],
    names: tuple[str, ...],
    minimum_visibility: float,
) -> tuple[Landmark, ...] | None:
    points = tuple(landmarks.get(name) for name in names)
    if any(not _visible(point, minimum_visibility) for point in points):
        return None
    return points  # type: ignore[return-value]


def _visible(point: Landmark | None, minimum_visibility: float) -> bool:
    return point is not None and point.visibility >= minimum_visibility


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None

