"""Bounded time windows and compact session evidence for the existing rule engine."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import ceil

from care_hero.schemas import FaceMeshObservation, Keypoint, VideoKeypointResult, VideoMetadata
from care_hero.scoring import score_keypoints


class OnlineMetrics:
    def __init__(self, settings, rules) -> None:
        self.settings = settings
        self.rules = rules
        # This limit also protects against a source that ignores the requested Hz.
        self._session = deque(maxlen=ceil(settings.duration_seconds * settings.analysis_hz) + 2)
        self._window = deque()
        self.frame_count = 0
        self.face_detected = 0
        self.face_errors = 0

    def update(self, frame, width: int, height: int) -> None:
        self.frame_count += 1
        self.face_detected += frame.face_mesh.detected
        self.face_errors += frame.face_mesh.status == "error"
        # x/y normalized by different axes distort angles on non-square images.
        # Use image-height units for both axes, retaining original frame separately.
        ratio = width / height
        compact = replace(
            frame,
            landmarks={name: Keypoint(p.x * ratio, p.y, p.visibility) for name, p in frame.landmarks.items()},
            face_mesh=FaceMeshObservation(status=frame.face_mesh.status),
        )
        self._session.append(compact)
        self._window.append(compact)
        cutoff = frame.timestamp_seconds - self.settings.window_seconds
        while self._window and self._window[0].timestamp_seconds < cutoff:
            self._window.popleft()

    def _result(self, frames) -> VideoKeypointResult:
        items = list(frames)
        duration = items[-1].timestamp_seconds - items[0].timestamp_seconds if items else 0.0
        return VideoKeypointResult(VideoMetadata(self.settings.analysis_hz, len(items), duration, 0, 0), items)

    def snapshot(self) -> dict:
        result = self._result(self._window)
        return {
            "window_start_seconds": result.frames[0].timestamp_seconds if result.frames else None,
            "window_end_seconds": result.frames[-1].timestamp_seconds if result.frames else None,
            "analyzed_frame_count": result.analyzed_frame_count,
            "rule_based": score_keypoints(result, self.rules).to_dict(),
        }

    def final_report(self):
        return score_keypoints(self._result(self._session), self.rules)

    def summary(self) -> dict:
        return {
            "analyzed_frames": self.frame_count,
            "retained_compact_frames": len(self._session),
            "face_detected_frames": self.face_detected,
            "face_error_frames": self.face_errors,
            "coordinate_space_for_scoring": "image_height_normalized",
        }
