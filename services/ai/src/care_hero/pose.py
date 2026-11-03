"""MediaPipe Pose lifecycle and per-frame keypoint extraction."""

from __future__ import annotations

import statistics
from contextlib import contextmanager
from typing import Any, Iterator

from care_hero.config import AppConfig, load_config
from care_hero.errors import DependencyError
from care_hero.schemas import FrameKeypoints, Keypoint


@contextmanager
def load_mp_pose(config: AppConfig | None = None) -> Iterator[Any]:
    """Initialize MediaPipe Pose once per image/video processing session."""

    settings = config or load_config()
    try:
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=settings.pose.model_complexity,
            smooth_landmarks=settings.pose.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=settings.pose.min_detection_confidence,
            min_tracking_confidence=settings.pose.min_tracking_confidence,
        )
    except (ImportError, OSError, AttributeError, RuntimeError) as exc:
        raise DependencyError("MediaPipe Pose를 초기화할 수 없습니다.") from exc
    try:
        yield pose
    finally:
        pose.close()


def extract_keypoints(
    bgr_frame: Any,
    mp_pose: Any,
    frame_index: int,
    timestamp_seconds: float,
) -> FrameKeypoints:
    """Run one BGR frame through MediaPipe and return all named pose landmarks."""

    try:
        import cv2
        import mediapipe as mp
    except (ImportError, OSError) as exc:
        raise DependencyError("MediaPipe 또는 OpenCV를 불러올 수 없습니다.") from exc

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    result = mp_pose.process(rgb_frame)
    if result.pose_landmarks is None:
        return FrameKeypoints(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            detected=False,
        )

    landmarks: dict[str, Keypoint] = {}
    for landmark_type in mp.solutions.pose.PoseLandmark:
        landmark = result.pose_landmarks.landmark[landmark_type.value]
        landmarks[landmark_type.name.lower()] = Keypoint(
            x=float(landmark.x),
            y=float(landmark.y),
            visibility=float(landmark.visibility),
        )
    visibility = [point.visibility for point in landmarks.values()]
    return FrameKeypoints(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        detected=True,
        landmarks=landmarks,
        mean_visibility=statistics.fmean(visibility) if visibility else None,
    )

