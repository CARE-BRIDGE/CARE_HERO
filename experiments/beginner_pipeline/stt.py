"""선택 기능: faster-whisper를 사용해 영상의 발화 내용과 시간을 전사한다."""

from __future__ import annotations

from pathlib import Path


def transcribe_video(video_path: str | Path, model_size: str = "small") -> dict[str, object]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "STT를 사용하려면 `pip install faster-whisper`를 먼저 실행해 주세요."
        ) from exc

    # CPU에서도 실행 가능하도록 int8을 기본으로 사용한다.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), language="ko", vad_filter=True)
    transcript_segments = [
        {
            "start_seconds": round(float(segment.start), 3),
            "end_seconds": round(float(segment.end), 3),
            "text": segment.text.strip(),
        }
        for segment in segments
    ]
    return {
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "segments": transcript_segments,
        "full_text": " ".join(item["text"] for item in transcript_segments),
        "scope": "발화 내용 전사만 수행하며 목소리 감정·공격성·의도를 판단하지 않음",
    }

