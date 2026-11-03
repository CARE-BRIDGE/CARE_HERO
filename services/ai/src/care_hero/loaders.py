"""Streamlit-independent image/video loading and frame extraction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

from care_hero.config import AppConfig, load_config
from care_hero.errors import DependencyError, MediaReadError, UnsupportedMediaError
from care_hero.schemas import LoadedVideo, VideoFrame, VideoMetadata


def load_img(path: str | Path, config: AppConfig | None = None) -> Any:
    """Load one image as a resized OpenCV BGR array."""

    settings = config or load_config()
    image_path = Path(path)
    _validate_path(image_path, settings.video.supported_image_extensions, "이미지")
    cv2 = _cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MediaReadError("이미지 파일을 디코딩할 수 없습니다.")
    return _resize_frame(image, settings, cv2)


def load_video(path: str | Path, config: AppConfig | None = None) -> LoadedVideo:
    """Read video metadata and call `extract_frames` for lazy frame decoding."""

    settings = config or load_config()
    video_path = Path(path)
    _validate_path(video_path, settings.video.supported_extensions, "영상")
    metadata = _read_video_metadata(video_path)
    frames = extract_frames(video_path, metadata, settings)
    return LoadedVideo(metadata=metadata, frames=frames)


def extract_frames(
    path: str | Path,
    metadata: VideoMetadata,
    config: AppConfig | None = None,
) -> Iterator[VideoFrame]:
    """Yield sampled frames using timestamps from the original frame index."""

    settings = config or load_config()
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    decoded_frame_count = 0
    try:
        if not capture.isOpened():
            raise MediaReadError("프레임 추출을 위해 영상을 열 수 없습니다.")
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            decoded_frame_count += 1
            if frame_index % settings.video.sample_every_n_frames == 0:
                yield VideoFrame(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / metadata.fps,
                    bgr_frame=_resize_frame(frame, settings, cv2),
                )
            frame_index += 1
    finally:
        capture.release()
    if decoded_frame_count == 0:
        raise MediaReadError("영상 프레임을 디코딩할 수 없습니다.")


def _read_video_metadata(path: Path) -> VideoMetadata:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaReadError("영상 파일을 열 수 없습니다.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise MediaReadError("영상 FPS를 읽을 수 없습니다.")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise MediaReadError("영상 프레임 수 또는 해상도를 읽을 수 없습니다.")
    return VideoMetadata(
        fps=fps,
        total_frame_count=frame_count,
        duration_seconds=frame_count / fps,
        width=width,
        height=height,
    )


def _resize_frame(frame: Any, config: AppConfig, cv2: Any) -> Any:
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        config.video.max_analysis_width / width,
        config.video.max_analysis_height / height,
    )
    if scale >= 1.0:
        return frame
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def _validate_path(path: Path, extensions: tuple[str, ...], media_name: str) -> None:
    if path.suffix.lower() not in extensions:
        raise UnsupportedMediaError(f"지원하지 않는 {media_name} 형식입니다: {path.suffix}")
    if not path.is_file():
        raise MediaReadError(f"{media_name} 파일을 찾을 수 없습니다.")


def _cv2() -> Any:
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise DependencyError("OpenCV를 불러올 수 없습니다.") from exc
    return cv2

