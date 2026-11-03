from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from care_hero.errors import FaceMeshProcessingError
from care_hero.face_mesh import extract_face_landmarks, load_mp_face_mesh


class EmptyFaceMesh:
    def process(self, frame):
        return SimpleNamespace(multi_face_landmarks=None)


class DetectedFaceMesh:
    def process(self, frame):
        landmarks = [
            SimpleNamespace(x=0.1, y=0.2, z=-0.01),
            SimpleNamespace(x=0.3, y=0.4, z=-0.02),
        ]
        return SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
        )


class BrokenFaceMesh:
    def process(self, frame):
        raise RuntimeError("model failure")


def test_extract_face_landmarks_keeps_detection_failure_explicit() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = extract_face_landmarks(frame, EmptyFaceMesh())
    assert result.status == "detection_failed"
    assert not result.detected
    assert result.landmarks == []


def test_extract_face_landmarks_preserves_index_and_normalized_coordinates() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = extract_face_landmarks(frame, DetectedFaceMesh())
    assert result.status == "valid"
    assert result.detected
    assert len(result.landmarks) == 2
    assert result.landmarks[1].index == 1
    assert result.landmarks[1].x == pytest.approx(0.3)
    assert result.landmarks[1].z == pytest.approx(-0.02)


def test_extract_face_landmarks_wraps_frame_processing_errors() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(FaceMeshProcessingError):
        extract_face_landmarks(frame, BrokenFaceMesh())


def test_face_mesh_session_is_closed(monkeypatch, config) -> None:
    session = SimpleNamespace(closed=False)
    session.close = lambda: setattr(session, "closed", True)
    fake_constructor = lambda **kwargs: session
    fake_module = SimpleNamespace(
        solutions=SimpleNamespace(
            face_mesh=SimpleNamespace(FaceMesh=fake_constructor)
        )
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_module)
    with load_mp_face_mesh(config) as loaded:
        assert loaded is session
    assert session.closed
