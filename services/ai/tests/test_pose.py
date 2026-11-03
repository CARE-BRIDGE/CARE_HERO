from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from care_hero.pose import extract_keypoints


class EmptyPose:
    def process(self, frame):
        return SimpleNamespace(pose_landmarks=None)


def test_extract_keypoints_keeps_detection_failure_explicit() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = extract_keypoints(frame, EmptyPose(), 4, 0.2)
    assert result.frame_index == 4
    assert result.timestamp_seconds == pytest.approx(0.2)
    assert not result.detected
    assert result.landmarks == {}

