from __future__ import annotations

import json
from contextlib import contextmanager

from care_hero.pipeline import extract_video_keypoints, save_keypoints_json
from care_hero.schemas import (
    FaceLandmark,
    FaceMeshObservation,
    FrameKeypoints,
    Keypoint,
    LoadedVideo,
    VideoFrame,
    VideoMetadata,
)


def test_pipeline_uses_one_pose_session(monkeypatch, config) -> None:
    loaded = LoadedVideo(
        metadata=VideoMetadata(10.0, 2, 0.2, 320, 240),
        frames=iter([VideoFrame(0, 0.0, "a"), VideoFrame(1, 0.1, "b")]),
    )
    pose_sessions = []
    face_sessions = []

    @contextmanager
    def fake_pose(_):
        pose = object()
        pose_sessions.append(pose)
        yield pose

    @contextmanager
    def fake_face_mesh(_):
        face_mesh = object()
        face_sessions.append(face_mesh)
        yield face_mesh

    monkeypatch.setattr("care_hero.pipeline.load_video", lambda path, cfg: loaded)
    monkeypatch.setattr("care_hero.pipeline.load_mp_pose", fake_pose)
    monkeypatch.setattr("care_hero.pipeline.load_mp_face_mesh", fake_face_mesh)
    monkeypatch.setattr(
        "care_hero.pipeline.extract_keypoints",
        lambda frame, pose, index, timestamp: FrameKeypoints(
            index,
            timestamp,
            True,
            {"nose": Keypoint(0.5, 0.2, 0.9)},
            0.9,
        ),
    )
    monkeypatch.setattr(
        "care_hero.pipeline.extract_face_landmarks",
        lambda frame, face_mesh: FaceMeshObservation(
            status="valid",
            landmarks=[FaceLandmark(0, 0.5, 0.2, -0.01)],
        ),
    )
    result = extract_video_keypoints("sample.mp4", config)
    assert len(pose_sessions) == 1
    assert len(face_sessions) == 1
    assert result.analyzed_frame_count == 2
    assert result.detected_frame_count == 2
    assert all(frame.face_mesh.detected for frame in result.frames)


def test_pipeline_keeps_pose_result_when_face_mesh_frame_fails(monkeypatch, config) -> None:
    loaded = LoadedVideo(
        metadata=VideoMetadata(10.0, 1, 0.1, 320, 240),
        frames=iter([VideoFrame(0, 0.0, "frame")]),
    )

    @contextmanager
    def fake_pose(_):
        yield object()

    @contextmanager
    def fake_face_mesh(_):
        yield object()

    monkeypatch.setattr("care_hero.pipeline.load_video", lambda path, cfg: loaded)
    monkeypatch.setattr("care_hero.pipeline.load_mp_pose", fake_pose)
    monkeypatch.setattr("care_hero.pipeline.load_mp_face_mesh", fake_face_mesh)
    monkeypatch.setattr(
        "care_hero.pipeline.extract_keypoints",
        lambda frame, pose, index, timestamp: FrameKeypoints(
            index,
            timestamp,
            True,
            {"nose": Keypoint(0.5, 0.2, 0.9)},
            0.9,
        ),
    )

    def fail_face_mesh(frame, face_mesh):
        from care_hero.errors import FaceMeshProcessingError

        raise FaceMeshProcessingError("face error")

    monkeypatch.setattr("care_hero.pipeline.extract_face_landmarks", fail_face_mesh)
    result = extract_video_keypoints("sample.mp4", config)
    assert result.frames[0].detected
    assert result.frames[0].face_mesh.status == "error"


def test_pipeline_keeps_pose_result_when_face_mesh_initialization_fails(
    monkeypatch,
    config,
) -> None:
    from care_hero.errors import DependencyError

    loaded = LoadedVideo(
        metadata=VideoMetadata(10.0, 1, 0.1, 320, 240),
        frames=iter([VideoFrame(0, 0.0, "frame")]),
    )

    @contextmanager
    def fake_pose(_):
        yield object()

    @contextmanager
    def failed_face_mesh(_):
        raise DependencyError("face init error")
        yield

    monkeypatch.setattr("care_hero.pipeline.load_video", lambda path, cfg: loaded)
    monkeypatch.setattr("care_hero.pipeline.load_mp_pose", fake_pose)
    monkeypatch.setattr("care_hero.pipeline.load_mp_face_mesh", failed_face_mesh)
    monkeypatch.setattr(
        "care_hero.pipeline.extract_keypoints",
        lambda frame, pose, index, timestamp: FrameKeypoints(index, timestamp, True),
    )
    result = extract_video_keypoints("sample.mp4", config)
    assert result.frames[0].detected
    assert result.frames[0].face_mesh.status == "error"
    assert result.frames[0].face_mesh.error == "face init error"


def test_pipeline_skips_face_mesh_when_disabled(monkeypatch, config) -> None:
    from dataclasses import replace

    disabled = replace(config, face_mesh=replace(config.face_mesh, enabled=False))
    loaded = LoadedVideo(
        metadata=VideoMetadata(10.0, 1, 0.1, 320, 240),
        frames=iter([VideoFrame(0, 0.0, "frame")]),
    )

    @contextmanager
    def fake_pose(_):
        yield object()

    monkeypatch.setattr("care_hero.pipeline.load_video", lambda path, cfg: loaded)
    monkeypatch.setattr("care_hero.pipeline.load_mp_pose", fake_pose)
    monkeypatch.setattr(
        "care_hero.pipeline.extract_keypoints",
        lambda frame, pose, index, timestamp: FrameKeypoints(index, timestamp, False),
    )
    result = extract_video_keypoints("sample.mp4", disabled)
    assert result.frames[0].face_mesh.status == "not_run"


def test_save_keypoints_json(tmp_path) -> None:
    result = FrameKeypoints(0, 0.0, False)
    from care_hero.schemas import VideoKeypointResult

    video_result = VideoKeypointResult(VideoMetadata(30.0, 1, 1 / 30, 640, 360), [result])
    output = save_keypoints_json(video_result, tmp_path / "result.json")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["summary"]["detected_frame_count"] == 0
    assert data["summary"]["body_coverage"]["profile"] == "detection_failed"
    assert data["frames"][0]["detected"] is False
    assert data["frames"][0]["body_coverage"] == "detection_failed"
    assert data["frames"][0]["face_mesh"]["status"] == "not_run"
    assert data["summary"]["face_mesh"]["detection_rate"] is None
