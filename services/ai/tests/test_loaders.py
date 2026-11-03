from __future__ import annotations

from pathlib import Path

import pytest

from care_hero.errors import MediaReadError, UnsupportedMediaError
from care_hero.loaders import load_video
from care_hero.schemas import VideoFrame, VideoMetadata


def test_load_video_calls_extract_frames(monkeypatch, tmp_path, config) -> None:
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"placeholder")
    metadata = VideoMetadata(30.0, 30, 1.0, 640, 360)
    frames = iter([VideoFrame(0, 0.0, "frame")])
    calls: list[tuple[Path, VideoMetadata]] = []
    monkeypatch.setattr("care_hero.loaders._read_video_metadata", lambda _: metadata)

    def fake_extract(received_path, received_metadata, received_config):
        calls.append((Path(received_path), received_metadata))
        return frames

    monkeypatch.setattr("care_hero.loaders.extract_frames", fake_extract)
    loaded = load_video(path, config)
    assert loaded.metadata is metadata
    assert list(loaded.frames)[0].bgr_frame == "frame"
    assert calls == [(path, metadata)]


def test_load_video_rejects_missing_file(tmp_path, config) -> None:
    with pytest.raises(MediaReadError):
        load_video(tmp_path / "missing.mp4", config)


def test_load_video_rejects_extension(tmp_path, config) -> None:
    path = tmp_path / "sample.mkv"
    path.write_bytes(b"placeholder")
    with pytest.raises(UnsupportedMediaError):
        load_video(path, config)

