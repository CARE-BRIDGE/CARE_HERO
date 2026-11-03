"""Local video -> MediaPipe keypoint -> JSON orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

from care_hero.config import AppConfig, load_config
from care_hero.errors import DependencyError, FaceMeshProcessingError
from care_hero.face_mesh import extract_face_landmarks, load_mp_face_mesh
from care_hero.loaders import load_video
from care_hero.pose import extract_keypoints, load_mp_pose
from care_hero.schemas import FaceMeshObservation, VideoKeypointResult


ProgressCallback = Callable[[int, int], None]


def extract_video_keypoints(
    video_path: str | Path,
    config: AppConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VideoKeypointResult:
    """Extract keypoints while keeping only the current image frame in memory."""

    settings = config or load_config()
    loaded = load_video(video_path, settings)
    expected = max(
        1,
        (
            loaded.metadata.total_frame_count
            + settings.video.sample_every_n_frames
            - 1
        )
        // settings.video.sample_every_n_frames,
    )
    frame_results = []
    with load_mp_pose(settings) as mp_pose, ExitStack() as face_stack:
        face_mesh = None
        face_initialization_error = None
        if settings.face_mesh.enabled:
            try:
                face_mesh = face_stack.enter_context(load_mp_face_mesh(settings))
            except DependencyError as exc:
                face_initialization_error = str(exc)
        for processed_count, frame in enumerate(loaded.frames, start=1):
            pose_result = extract_keypoints(
                frame.bgr_frame,
                mp_pose,
                frame.frame_index,
                frame.timestamp_seconds,
            )
            pose_result.face_mesh = _extract_optional_face_mesh(
                frame.bgr_frame,
                face_mesh,
                enabled=settings.face_mesh.enabled,
                initialization_error=face_initialization_error,
            )
            frame_results.append(pose_result)
            if progress_callback is not None:
                progress_callback(processed_count, expected)
    return VideoKeypointResult(
        video_metadata=loaded.metadata,
        frames=frame_results,
    )


def _extract_optional_face_mesh(
    bgr_frame: object,
    face_mesh: object | None,
    *,
    enabled: bool,
    initialization_error: str | None,
) -> FaceMeshObservation:
    if not enabled:
        return FaceMeshObservation(status="not_run")
    if face_mesh is None:
        return FaceMeshObservation(
            status="error",
            error=initialization_error or "FaceMesh를 초기화하지 못했습니다.",
        )
    try:
        return extract_face_landmarks(bgr_frame, face_mesh)
    except (DependencyError, FaceMeshProcessingError) as exc:
        return FaceMeshObservation(status="error", error=str(exc))


def save_keypoints_json(result: VideoKeypointResult, output_path: str | Path) -> Path:
    """Persist structured keypoints as UTF-8 JSON for inspection and later LLM input."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
