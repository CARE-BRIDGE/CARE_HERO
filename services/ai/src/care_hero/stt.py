"""Optional local speech-to-text using an injectable backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from care_hero.config import STTConfig
from care_hero.errors import SpeechTranscriptionError
from care_hero.schemas import TranscriptSegment, TranscriptionResult


class SpeechTranscriber(Protocol):
    def transcribe(self, media_path: Path) -> TranscriptionResult:
        """Transcribe one local media file."""


class FasterWhisperTranscriber:
    """Load faster-whisper once and return a typed local transcript."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        try:
            from faster_whisper import WhisperModel
        except (ImportError, OSError) as exc:
            raise SpeechTranscriptionError(
                "STT 의존성이 없습니다. `pip install -e '.[stt]'`를 실행하세요."
            ) from exc
        try:
            self.model: Any = WhisperModel(
                config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SpeechTranscriptionError("로컬 STT 모델을 초기화할 수 없습니다.") from exc

    def transcribe(self, media_path: Path) -> TranscriptionResult:
        try:
            segment_iter, info = self.model.transcribe(
                str(media_path),
                language=self.config.language,
                vad_filter=self.config.vad_filter,
            )
            segments = [
                TranscriptSegment(
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=str(segment.text).strip(),
                )
                for segment in segment_iter
                if str(segment.text).strip()
            ]
        except (OSError, RuntimeError, ValueError) as exc:
            raise SpeechTranscriptionError("음성 전사 중 오류가 발생했습니다.") from exc
        duration = max((segment.end_seconds for segment in segments), default=None)
        return TranscriptionResult(
            status="success",
            backend="faster-whisper",
            model=self.config.model_size,
            language=getattr(info, "language", self.config.language),
            duration_seconds=duration,
            segments=segments,
        )


def transcribe_media(
    media_path: str | Path,
    config: STTConfig,
    transcriber: SpeechTranscriber | None = None,
) -> TranscriptionResult:
    """Transcribe local media without coupling STT to pose analysis."""

    path = Path(media_path)
    if not path.is_file():
        raise SpeechTranscriptionError("음성 전사 대상 파일을 찾을 수 없습니다.")
    backend = transcriber or FasterWhisperTranscriber(config)
    return backend.transcribe(path)


def save_transcription_json(
    result: TranscriptionResult,
    output_path: str | Path,
) -> Path:
    """Persist transcript atomically without printing its contents."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
