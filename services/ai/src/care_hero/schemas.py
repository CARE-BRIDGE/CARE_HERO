"""Typed values exchanged by the local extraction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    total_frame_count: int
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    timestamp_seconds: float
    bgr_frame: Any


@dataclass(frozen=True)
class LoadedVideo:
    metadata: VideoMetadata
    frames: Iterator[VideoFrame]


@dataclass(frozen=True)
class Keypoint:
    """MediaPipe normalized coordinates and detector visibility.

    The x/y values are image-relative observations, not physical coordinates.
    """

    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class FaceLandmark:
    """Image-relative FaceMesh coordinate, not a physical 3D position."""

    index: int
    x: float
    y: float
    z: float


@dataclass
class FaceMeshObservation:
    status: str = "not_run"
    landmarks: list[FaceLandmark] = field(default_factory=list)
    error: str | None = None

    @property
    def detected(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detected": self.detected,
            "landmark_count": len(self.landmarks),
            "landmarks": [asdict(point) for point in self.landmarks],
            "error": self.error,
        }


@dataclass
class FrameKeypoints:
    frame_index: int
    timestamp_seconds: float
    detected: bool
    landmarks: dict[str, Keypoint] = field(default_factory=dict)
    mean_visibility: float | None = None
    face_mesh: FaceMeshObservation = field(default_factory=FaceMeshObservation)

    def to_dict(self) -> dict[str, Any]:
        from care_hero.coverage import classify_frame_coverage

        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "detected": self.detected,
            "body_coverage": classify_frame_coverage(
                detected=self.detected,
                landmarks=self.landmarks,
            ),
            "landmarks": {
                name: asdict(point) for name, point in self.landmarks.items()
            },
            "mean_visibility": self.mean_visibility,
            "face_mesh": self.face_mesh.to_dict(),
        }


@dataclass
class VideoKeypointResult:
    video_metadata: VideoMetadata
    frames: list[FrameKeypoints]

    @property
    def analyzed_frame_count(self) -> int:
        return len(self.frames)

    @property
    def detected_frame_count(self) -> int:
        return sum(frame.detected for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        from care_hero.coverage import summarize_body_coverage

        analyzed = self.analyzed_frame_count
        coverage = summarize_body_coverage(self.frames)
        face_observations = [
            frame.face_mesh
            for frame in self.frames
            if frame.face_mesh.status != "not_run"
        ]
        face_detected = sum(observation.detected for observation in face_observations)
        face_status_counts: dict[str, int] = {}
        for observation in face_observations:
            face_status_counts[observation.status] = (
                face_status_counts.get(observation.status, 0) + 1
            )
        return {
            "video_metadata": asdict(self.video_metadata),
            "summary": {
                "analyzed_frame_count": analyzed,
                "detected_frame_count": self.detected_frame_count,
                "detection_rate": (
                    self.detected_frame_count / analyzed if analyzed else 0.0
                ),
                "body_coverage": coverage.to_dict(),
                "face_mesh": {
                    "analyzed_frame_count": len(face_observations),
                    "detected_frame_count": face_detected,
                    "detection_rate": (
                        face_detected / len(face_observations)
                        if face_observations
                        else None
                    ),
                    "status_counts": face_status_counts,
                },
            },
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass
class RuleMetricResult:
    metric_id: str
    display_name: str
    raw_value: float | None
    unit: str
    status: str
    score: float | None
    weight: float
    valid_frame_count: int
    total_frame_count: int
    feedback: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleBasedReport:
    scenario_id: str
    scenario_name: str
    threshold_status: str
    overall_status: str
    total_score: float | None
    metrics: list[RuleMetricResult]
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "threshold_status": self.threshold_status,
            "overall_status": self.overall_status,
            "total_score": self.total_score,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptionResult:
    status: str
    backend: str
    model: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    error: str | None = None

    @property
    def full_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "model": self.model,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "full_text": self.full_text,
            "segments": [segment.to_dict() for segment in self.segments],
            "error": self.error,
        }


@dataclass
class FinalAnalysisReport:
    schema_version: str
    generated_at: str
    video_summary: dict[str, Any]
    rule_based: RuleBasedReport
    llm_status: str
    llm_feedback: dict[str, Any] | None
    llm_error: str | None = None
    transcription: TranscriptionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "video_summary": self.video_summary,
            "rule_based": self.rule_based.to_dict(),
            "llm_status": self.llm_status,
            "llm_feedback": self.llm_feedback,
            "llm_error": self.llm_error,
            "transcription": (
                self.transcription.to_dict() if self.transcription is not None else None
            ),
        }
