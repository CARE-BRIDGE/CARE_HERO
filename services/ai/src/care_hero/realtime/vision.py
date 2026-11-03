"""Reuse existing Pose/FaceMesh adapters and expose explicit per-frame timings."""

from __future__ import annotations

from contextlib import ExitStack
from time import perf_counter

from care_hero.errors import DependencyError, FaceMeshProcessingError
from care_hero.face_mesh import extract_face_landmarks, load_mp_face_mesh
from care_hero.loaders import _resize_frame
from care_hero.pose import extract_keypoints, load_mp_pose
from care_hero.schemas import FaceMeshObservation


class VisionAnalyzer:
    """Initialize and use MediaPipe on its owning (consumer) thread."""

    def __init__(self, config) -> None:
        self.config = config
        self._stack = ExitStack()
        self.pose = None
        self.face = None

    def __enter__(self):
        try:
            self.pose = self._stack.enter_context(load_mp_pose(self.config))
            if self.config.face_mesh.enabled:
                try:
                    self.face = self._stack.enter_context(load_mp_face_mesh(self.config))
                except DependencyError:
                    self.face = None
        except Exception:
            self._stack.close()
            raise
        return self

    def analyze(self, packet):
        import cv2

        image = _resize_frame(packet.image, self.config, cv2)
        before = perf_counter()
        result = extract_keypoints(image, self.pose, packet.index, packet.timestamp_seconds)
        pose_ms = (perf_counter() - before) * 1000
        before = perf_counter()
        if not self.config.face_mesh.enabled:
            result.face_mesh = FaceMeshObservation(status="not_run")
        elif self.face is None:
            result.face_mesh = FaceMeshObservation(status="error", error="face_initialization_failed")
        else:
            try:
                result.face_mesh = extract_face_landmarks(image, self.face)
            except (DependencyError, FaceMeshProcessingError):
                result.face_mesh = FaceMeshObservation(status="error", error="face_processing_failed")
        return result, image, {"pose_ms": pose_ms, "face_ms": (perf_counter() - before) * 1000}

    def __exit__(self, *args):
        self._stack.close()


def draw_preview(image, frame, status_text: str):
    """Display only valid landmarks; skip off-frame/low-confidence connections."""
    import cv2
    import mediapipe as mp

    output = image.copy()
    height, width = output.shape[:2]
    points = frame.landmarks
    names = [item.name.lower() for item in mp.solutions.pose.PoseLandmark]

    def visible(point):
        return point is not None and point.visibility >= 0.5 and 0 <= point.x <= 1 and 0 <= point.y <= 1

    for first, second in mp.solutions.pose.POSE_CONNECTIONS:
        a, b = points.get(names[first]), points.get(names[second])
        if visible(a) and visible(b):
            cv2.line(output, (round(a.x * width), round(a.y * height)),
                     (round(b.x * width), round(b.y * height)), (170, 190, 20), 2)
    face = frame.face_mesh.landmarks
    for first, second in mp.solutions.face_mesh.FACEMESH_CONTOURS:
        if max(first, second) < len(face):
            a, b = face[first], face[second]
            cv2.line(output, (round(a.x * width), round(a.y * height)),
                     (round(b.x * width), round(b.y * height)), (140, 190, 50), 1)
    cv2.putText(output, status_text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 230, 230), 2)
    return output
