"""YAML configuration loader for the local pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from care_hero.errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


@dataclass(frozen=True)
class VideoConfig:
    supported_extensions: tuple[str, ...]
    supported_image_extensions: tuple[str, ...]
    max_analysis_width: int
    max_analysis_height: int
    sample_every_n_frames: int


@dataclass(frozen=True)
class PoseConfig:
    model_complexity: int
    min_detection_confidence: float
    min_tracking_confidence: float
    smooth_landmarks: bool


@dataclass(frozen=True)
class FaceMeshConfig:
    enabled: bool
    max_num_faces: int
    refine_landmarks: bool
    min_detection_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class STTConfig:
    enabled: bool
    model_size: str
    device: str
    compute_type: str
    language: str | None
    vad_filter: bool


@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig
    pose: PoseConfig
    face_mesh: FaceMeshConfig
    stt: STTConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError("설정 파일을 찾을 수 없습니다.")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("설정 YAML을 읽을 수 없습니다.") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("설정의 최상위 값은 객체여야 합니다.")
    video = _mapping(raw.get("video"), "video")
    pose = _mapping(raw.get("pose"), "pose")
    face_mesh = _mapping(raw.get("face_mesh"), "face_mesh")
    stt = _mapping(raw.get("stt"), "stt")

    model_complexity = pose.get("model_complexity")
    if model_complexity not in (0, 1, 2):
        raise ConfigurationError("pose.model_complexity는 0, 1, 2 중 하나여야 합니다.")
    max_num_faces = _positive_int(
        face_mesh.get("max_num_faces"),
        "face_mesh.max_num_faces",
    )
    if max_num_faces != 1:
        raise ConfigurationError("현재 파이프라인은 얼굴 한 명만 처리합니다.")

    return AppConfig(
        video=VideoConfig(
            supported_extensions=_extensions(video.get("supported_extensions"), "video.supported_extensions"),
            supported_image_extensions=_extensions(
                video.get("supported_image_extensions"),
                "video.supported_image_extensions",
            ),
            max_analysis_width=_positive_int(video.get("max_analysis_width"), "video.max_analysis_width"),
            max_analysis_height=_positive_int(video.get("max_analysis_height"), "video.max_analysis_height"),
            sample_every_n_frames=_positive_int(
                video.get("sample_every_n_frames"),
                "video.sample_every_n_frames",
            ),
        ),
        pose=PoseConfig(
            model_complexity=model_complexity,
            min_detection_confidence=_probability(
                pose.get("min_detection_confidence"),
                "pose.min_detection_confidence",
            ),
            min_tracking_confidence=_probability(
                pose.get("min_tracking_confidence"),
                "pose.min_tracking_confidence",
            ),
            smooth_landmarks=_boolean(pose.get("smooth_landmarks"), "pose.smooth_landmarks"),
        ),
        face_mesh=FaceMeshConfig(
            enabled=_boolean(face_mesh.get("enabled"), "face_mesh.enabled"),
            max_num_faces=max_num_faces,
            refine_landmarks=_boolean(
                face_mesh.get("refine_landmarks"),
                "face_mesh.refine_landmarks",
            ),
            min_detection_confidence=_probability(
                face_mesh.get("min_detection_confidence"),
                "face_mesh.min_detection_confidence",
            ),
            min_tracking_confidence=_probability(
                face_mesh.get("min_tracking_confidence"),
                "face_mesh.min_tracking_confidence",
            ),
        ),
        stt=STTConfig(
            enabled=_boolean(stt.get("enabled"), "stt.enabled"),
            model_size=_non_empty_string(stt.get("model_size"), "stt.model_size"),
            device=_non_empty_string(stt.get("device"), "stt.device"),
            compute_type=_non_empty_string(stt.get("compute_type"), "stt.compute_type"),
            language=_optional_string(stt.get("language"), "stt.language"),
            vad_filter=_boolean(stt.get("vad_filter"), "stt.vad_filter"),
        ),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} 설정은 객체여야 합니다.")
    return value


def _extensions(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"{label} 설정은 문자열 목록이어야 합니다.")
    return tuple(item.lower() if item.startswith(".") else f".{item.lower()}" for item in value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{label} 설정은 0보다 큰 정수여야 합니다.")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigurationError(f"{label} 설정은 0과 1 사이여야 합니다.")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{label} 설정은 true 또는 false여야 합니다.")
    return value


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{label} 설정은 비어 있지 않은 문자열이어야 합니다.")
    return value.strip()


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, label)
