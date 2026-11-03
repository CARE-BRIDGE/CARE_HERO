"""Transparent provisional rule-based scoring from extracted pose keypoints."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from care_hero.coverage import summarize_body_coverage
from care_hero.errors import ScoringError
from care_hero.schemas import (
    FrameKeypoints,
    Keypoint,
    RuleBasedReport,
    RuleMetricResult,
    VideoKeypointResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_PATH = PROJECT_ROOT / "configs" / "scenarios" / "dementia_first_visit.yaml"
KNOWN_METRICS = {
    "detection_quality",
    "front_facing_ratio",
    "torso_alignment",
    "torso_stability",
}


@dataclass(frozen=True)
class MetricRule:
    id: str
    display_name: str
    enabled: bool
    weight: float
    evaluation: str
    unit: str
    feedback: dict[str, str]
    target: float | None = None
    good_maximum: float | None = None
    poor_minimum: float | None = None
    maximum_nose_offset_shoulder_widths: float | None = None


@dataclass(frozen=True)
class ScenarioRules:
    scenario_id: str
    scenario_name: str
    description: str
    threshold_status: str
    minimum_valid_frames: int
    minimum_detection_rate: float
    minimum_landmark_visibility: float
    good_score_threshold: float
    metrics: tuple[MetricRule, ...]


def load_scenario_rules(path: str | Path = DEFAULT_SCENARIO_PATH) -> ScenarioRules:
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise ScoringError("시나리오 점수 설정 파일을 찾을 수 없습니다.")
    try:
        raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScoringError("시나리오 점수 YAML을 읽을 수 없습니다.") from exc
    if not isinstance(raw, dict):
        raise ScoringError("시나리오 점수 설정은 객체여야 합니다.")
    raw_metrics = raw.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ScoringError("시나리오 metrics가 비어 있습니다.")

    metrics: list[MetricRule] = []
    for item in raw_metrics:
        if not isinstance(item, dict) or item.get("id") not in KNOWN_METRICS:
            raise ScoringError(f"알 수 없는 점수 Metric입니다: {item.get('id') if isinstance(item, dict) else item}")
        feedback = item.get("feedback")
        if not isinstance(feedback, dict) or not all(
            isinstance(feedback.get(key), str) for key in ("good", "improve", "unavailable")
        ):
            raise ScoringError(f"{item['id']} feedback 템플릿이 잘못되었습니다.")
        metrics.append(
            MetricRule(
                id=item["id"],
                display_name=_required_string(item, "display_name"),
                enabled=_required_bool(item, "enabled"),
                weight=_number(item, "weight"),
                evaluation=_required_string(item, "evaluation"),
                unit=_required_string(item, "unit"),
                feedback=feedback,
                target=_optional_number(item, "target"),
                good_maximum=_optional_number(item, "good_maximum"),
                poor_minimum=_optional_number(item, "poor_minimum"),
                maximum_nose_offset_shoulder_widths=_optional_number(
                    item, "maximum_nose_offset_shoulder_widths"
                ),
            )
        )
    enabled_weight = sum(metric.weight for metric in metrics if metric.enabled)
    if not math.isclose(enabled_weight, 1.0, abs_tol=1e-9):
        raise ScoringError("활성 Metric 가중치 합은 1.0이어야 합니다.")
    return ScenarioRules(
        scenario_id=_required_string(raw, "scenario_id"),
        scenario_name=_required_string(raw, "scenario_name"),
        description=_required_string(raw, "description"),
        threshold_status=_required_string(raw, "threshold_status"),
        minimum_valid_frames=_positive_int(raw, "minimum_valid_frames"),
        minimum_detection_rate=_probability(raw, "minimum_detection_rate"),
        minimum_landmark_visibility=_probability(raw, "minimum_landmark_visibility"),
        good_score_threshold=_percentage(raw, "good_score_threshold"),
        metrics=tuple(metrics),
    )


def score_keypoints(
    result: VideoKeypointResult,
    rules: ScenarioRules | None = None,
) -> RuleBasedReport:
    """Calculate provisional rule scores without using an LLM."""

    scenario = rules or load_scenario_rules()
    total = result.analyzed_frame_count
    detected = result.detected_frame_count
    detection_rate = detected / total if total else 0.0
    detection_status = _data_status(
        detected,
        total,
        scenario.minimum_valid_frames,
        scenario.minimum_detection_rate,
    )

    tilt_values = _torso_tilts(result.frames, scenario.minimum_landmark_visibility)
    front_values = _front_facing_flags(
        result.frames,
        scenario.minimum_landmark_visibility,
        _front_criterion(scenario),
    )
    coverage = summarize_body_coverage(
        result.frames,
        minimum_visibility=scenario.minimum_landmark_visibility,
    )
    torso_alignment = _series_measurement(
        [abs(value) for value in tilt_values],
        scenario.minimum_valid_frames,
    )
    torso_stability = _series_measurement(
        tilt_values,
        scenario.minimum_valid_frames,
        reducer=statistics.pstdev,
    )
    if (
        coverage.profile == "upper_body"
        and len(tilt_values) < scenario.minimum_valid_frames
    ):
        upper_body_unavailable = (
            None,
            0,
            "not_applicable",
            ["상반신 촬영으로 골반 landmark가 충분하지 않아 이 지표를 계산하지 않습니다."],
        )
        torso_alignment = upper_body_unavailable
        torso_stability = upper_body_unavailable
    measurements: dict[str, tuple[float | None, int, str, list[str]]] = {
        "detection_quality": (
            detection_rate if total else None,
            detected,
            detection_status,
            [] if detection_status == "valid" else ["검출률 또는 검출 프레임 수가 최소 기준보다 낮습니다."],
        ),
        "front_facing_ratio": _series_measurement(front_values, scenario.minimum_valid_frames),
        "torso_alignment": torso_alignment,
        "torso_stability": torso_stability,
    }

    metric_results: list[RuleMetricResult] = []
    for rule in scenario.metrics:
        if not rule.enabled:
            continue
        raw_value, valid_count, status, warnings = measurements[rule.id]
        score = _calculate_score(raw_value, status, rule)
        feedback_key = (
            "unavailable"
            if status != "valid" or score is None
            else "good"
            if score >= scenario.good_score_threshold
            else "improve"
        )
        metric_results.append(
            RuleMetricResult(
                metric_id=rule.id,
                display_name=rule.display_name,
                raw_value=raw_value,
                unit=rule.unit,
                status=status,
                score=score,
                weight=rule.weight,
                valid_frame_count=valid_count,
                total_frame_count=total,
                feedback=rule.feedback[feedback_key],
                warnings=warnings,
            )
        )

    all_valid = bool(metric_results) and all(metric.status == "valid" for metric in metric_results)
    partial = (
        any(metric.status == "valid" for metric in metric_results)
        and all(metric.status in {"valid", "not_applicable"} for metric in metric_results)
    )
    total_score = (
        round(
            sum((metric.score or 0.0) * metric.weight for metric in metric_results),
            2,
        )
        if all_valid
        else None
    )
    caveats = [
        "모든 임계값과 점수는 provisional이며 공식 교육 기준이 아닙니다.",
        "2D MediaPipe landmark 기반 결과로 실제 거리, 속도, 시선 또는 의도를 측정하지 않습니다.",
    ]
    if not all_valid:
        caveats.append("필수 데이터가 부족해 가중 총점을 생성하지 않았습니다.")
    if coverage.profile == "upper_body":
        caveats.append("상반신 촬영으로 확인 가능한 landmark 기반 지표만 평가했습니다.")
    return RuleBasedReport(
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.scenario_name,
        threshold_status=scenario.threshold_status,
        overall_status="valid" if all_valid else "partial" if partial else "insufficient_data",
        total_score=total_score,
        metrics=metric_results,
        caveats=caveats,
    )


def _data_status(
    valid_count: int,
    total_count: int,
    minimum_valid_frames: int,
    minimum_detection_rate: float,
) -> str:
    if total_count == 0 or valid_count == 0:
        return "detection_failed"
    if valid_count < minimum_valid_frames or valid_count / total_count < minimum_detection_rate:
        return "insufficient_data"
    return "valid"


def _series_measurement(
    values: list[float],
    minimum_valid_frames: int,
    reducer: Any = statistics.fmean,
) -> tuple[float | None, int, str, list[str]]:
    if not values:
        return None, 0, "detection_failed", ["필수 landmark를 검출하지 못했습니다."]
    if len(values) < minimum_valid_frames:
        return None, len(values), "insufficient_data", ["유효 landmark 프레임 수가 부족합니다."]
    return float(reducer(values)), len(values), "valid", []


def _front_facing_flags(
    frames: list[FrameKeypoints],
    minimum_visibility: float,
    maximum_offset: float,
) -> list[float]:
    flags: list[float] = []
    for frame in frames:
        points = _required_points(
            frame,
            ("nose", "left_shoulder", "right_shoulder"),
            minimum_visibility,
        )
        if points is None:
            continue
        nose, left, right = points
        center_x = (left.x + right.x) / 2.0
        width = math.hypot(right.x - left.x, right.y - left.y)
        if width <= 1e-9:
            continue
        flags.append(1.0 if abs(nose.x - center_x) / width <= maximum_offset else 0.0)
    return flags


def _torso_tilts(frames: list[FrameKeypoints], minimum_visibility: float) -> list[float]:
    values: list[float] = []
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    for frame in frames:
        points = _required_points(frame, names, minimum_visibility)
        if points is None:
            continue
        left_shoulder, right_shoulder, left_hip, right_hip = points
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        dx = shoulder_x - hip_x
        dy_up = hip_y - shoulder_y
        if math.hypot(dx, dy_up) <= 1e-9:
            continue
        values.append(math.degrees(math.atan2(dx, dy_up)))
    return values


def _required_points(
    frame: FrameKeypoints,
    names: tuple[str, ...],
    minimum_visibility: float,
) -> tuple[Keypoint, ...] | None:
    if not frame.detected:
        return None
    points = tuple(frame.landmarks.get(name) for name in names)
    if any(point is None or point.visibility < minimum_visibility for point in points):
        return None
    return points  # type: ignore[return-value]


def _calculate_score(raw_value: float | None, status: str, rule: MetricRule) -> float | None:
    if status != "valid" or raw_value is None:
        return None
    if rule.evaluation == "higher_is_better" and rule.target is not None and rule.target > 0:
        return round(max(0.0, min(100.0, raw_value / rule.target * 100.0)), 2)
    if (
        rule.evaluation == "lower_is_better"
        and rule.good_maximum is not None
        and rule.poor_minimum is not None
        and rule.poor_minimum > rule.good_maximum
    ):
        if raw_value <= rule.good_maximum:
            return 100.0
        if raw_value >= rule.poor_minimum:
            return 0.0
        fraction = (raw_value - rule.good_maximum) / (rule.poor_minimum - rule.good_maximum)
        return round((1.0 - fraction) * 100.0, 2)
    raise ScoringError(f"{rule.id}의 평가 설정이 올바르지 않습니다.")


def _front_criterion(rules: ScenarioRules) -> float:
    for metric in rules.metrics:
        if metric.id == "front_facing_ratio" and metric.maximum_nose_offset_shoulder_widths is not None:
            return metric.maximum_nose_offset_shoulder_widths
    raise ScoringError("front_facing_ratio 기준이 없습니다.")


def _required_string(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ScoringError(f"{key}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _required_bool(item: dict[str, Any], key: str) -> bool:
    value = item.get(key)
    if not isinstance(value, bool):
        raise ScoringError(f"{key}는 true 또는 false여야 합니다.")
    return value


def _number(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{key}는 숫자여야 합니다.")
    return float(value)


def _optional_number(item: dict[str, Any], key: str) -> float | None:
    return None if item.get(key) is None else _number(item, key)


def _positive_int(item: dict[str, Any], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScoringError(f"{key}는 0보다 큰 정수여야 합니다.")
    return value


def _probability(item: dict[str, Any], key: str) -> float:
    value = _number(item, key)
    if not 0 <= value <= 1:
        raise ScoringError(f"{key}는 0과 1 사이여야 합니다.")
    return value


def _percentage(item: dict[str, Any], key: str) -> float:
    value = _number(item, key)
    if not 0 <= value <= 100:
        raise ScoringError(f"{key}는 0과 100 사이여야 합니다.")
    return value
