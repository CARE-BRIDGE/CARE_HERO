from __future__ import annotations

from pathlib import Path

import pytest

from care_hero.schemas import FrameKeypoints, Keypoint, VideoKeypointResult, VideoMetadata
from care_hero.scoring import load_scenario_rules, score_keypoints


ROOT = Path(__file__).resolve().parents[1]


def pose_frame(index: int, *, nose_x: float = 0.5, tilt: float = 0.0) -> FrameKeypoints:
    visibility = 0.95
    return FrameKeypoints(
        frame_index=index,
        timestamp_seconds=index / 30,
        detected=True,
        landmarks={
            "nose": Keypoint(nose_x, 0.2, visibility),
            "left_shoulder": Keypoint(0.4 + tilt, 0.4, visibility),
            "right_shoulder": Keypoint(0.6 + tilt, 0.4, visibility),
            "left_hip": Keypoint(0.43 - tilt, 0.7, visibility),
            "right_hip": Keypoint(0.57 - tilt, 0.7, visibility),
        },
        mean_visibility=visibility,
    )


def upper_body_frame(index: int) -> FrameKeypoints:
    visibility = 0.95
    return FrameKeypoints(
        frame_index=index,
        timestamp_seconds=index / 30,
        detected=True,
        landmarks={
            "nose": Keypoint(0.5, 0.2, visibility),
            "left_shoulder": Keypoint(0.4, 0.4, visibility),
            "right_shoulder": Keypoint(0.6, 0.4, visibility),
            "left_elbow": Keypoint(0.35, 0.55, visibility),
            "right_elbow": Keypoint(0.65, 0.55, visibility),
        },
        mean_visibility=visibility,
    )


def test_rule_based_score_is_created_for_sufficient_data() -> None:
    frames = [pose_frame(index) for index in range(20)]
    result = VideoKeypointResult(VideoMetadata(30.0, 20, 2 / 3, 640, 360), frames)
    rules = load_scenario_rules(ROOT / "configs" / "scenarios" / "dementia_first_visit.yaml")
    report = score_keypoints(result, rules)
    assert report.overall_status == "valid"
    assert report.total_score == pytest.approx(100.0)
    assert all(metric.status == "valid" for metric in report.metrics)


def test_detection_failure_does_not_create_total_score() -> None:
    frames = [FrameKeypoints(index, index / 30, False) for index in range(20)]
    result = VideoKeypointResult(VideoMetadata(30.0, 20, 2 / 3, 640, 360), frames)
    report = score_keypoints(result, load_scenario_rules())
    assert report.overall_status == "insufficient_data"
    assert report.total_score is None
    assert all(metric.score is None for metric in report.metrics)


def test_front_facing_rule_uses_normalized_shoulder_width() -> None:
    frames = [pose_frame(index, nose_x=0.7) for index in range(20)]
    result = VideoKeypointResult(VideoMetadata(30.0, 20, 2 / 3, 1920, 1080), frames)
    report = score_keypoints(result, load_scenario_rules())
    front = next(metric for metric in report.metrics if metric.metric_id == "front_facing_ratio")
    assert front.raw_value == 0.0
    assert front.score == 0.0


def test_upper_body_video_returns_partial_result_without_fabricated_total() -> None:
    frames = [upper_body_frame(index) for index in range(20)]
    result = VideoKeypointResult(VideoMetadata(30.0, 20, 2 / 3, 640, 360), frames)
    report = score_keypoints(result, load_scenario_rules())
    statuses = {metric.metric_id: metric.status for metric in report.metrics}
    assert report.overall_status == "partial"
    assert report.total_score is None
    assert statuses["detection_quality"] == "valid"
    assert statuses["front_facing_ratio"] == "valid"
    assert statuses["torso_alignment"] == "not_applicable"
    assert statuses["torso_stability"] == "not_applicable"
