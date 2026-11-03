"""MediaPipe FaceMesh lifecycle and per-frame landmark extraction."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from care_hero.config import AppConfig, load_config
from care_hero.errors import DependencyError, FaceMeshProcessingError
from care_hero.schemas import FaceLandmark, FaceMeshObservation


@contextmanager
def load_mp_face_mesh(config: AppConfig | None = None) -> Iterator[Any]:
    """Initialize one FaceMesh tracker for an image/video processing session."""

    settings = config or load_config()
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=settings.face_mesh.max_num_faces,
            refine_landmarks=settings.face_mesh.refine_landmarks,
            min_detection_confidence=settings.face_mesh.min_detection_confidence,
            min_tracking_confidence=settings.face_mesh.min_tracking_confidence,
        )
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError) as exc:
        raise DependencyError("MediaPipe FaceMesh를 초기화할 수 없습니다.") from exc
    try:
        yield face_mesh
    finally:
        face_mesh.close()


def extract_face_landmarks(
    bgr_frame: Any,
    mp_face_mesh: Any,
) -> FaceMeshObservation:
    """Return the first detected face mesh without inferring emotion or intent."""

    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise DependencyError("FaceMesh 처리에 필요한 OpenCV를 불러올 수 없습니다.") from exc

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    try:
        result = mp_face_mesh.process(rgb_frame)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FaceMeshProcessingError("FaceMesh 프레임 처리에 실패했습니다.") from exc

    faces = getattr(result, "multi_face_landmarks", None)
    if not faces:
        return FaceMeshObservation(status="detection_failed")

    points = [
        FaceLandmark(
            index=index,
            x=float(landmark.x),
            y=float(landmark.y),
            z=float(landmark.z),
        )
        for index, landmark in enumerate(faces[0].landmark)
    ]
    return FaceMeshObservation(status="valid", landmarks=points)
