from __future__ import annotations

import json

from care_hero.config import load_config
from care_hero.schemas import TranscriptSegment, TranscriptionResult
from care_hero.stt import save_transcription_json, transcribe_media


class FakeTranscriber:
    def transcribe(self, media_path):
        return TranscriptionResult(
            status="success",
            backend="fake",
            model="test",
            language="ko",
            duration_seconds=1.5,
            segments=[TranscriptSegment(0.0, 1.5, "안녕하세요")],
        )


def test_transcription_backend_is_injectable(tmp_path) -> None:
    media = tmp_path / "sample.wav"
    media.write_bytes(b"test")
    result = transcribe_media(media, load_config().stt, FakeTranscriber())
    assert result.status == "success"
    assert result.full_text == "안녕하세요"

    output = save_transcription_json(result, tmp_path / "transcript.json")
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["language"] == "ko"
    assert saved["segments"][0]["start_seconds"] == 0.0
