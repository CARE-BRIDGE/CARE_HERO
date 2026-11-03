"""Energy endpointing and utterance-based (not token-streaming) local STT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from care_hero.realtime.capture import AudioBlock
from care_hero.realtime.settings import AudioSettings
from care_hero.schemas import TranscriptionResult, TranscriptSegment
from care_hero.stt import FasterWhisperTranscriber


@dataclass(frozen=True)
class Utterance:
    identifier: int
    start_seconds: float
    samples: np.ndarray
    sample_rate: int
    reason: str

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + len(self.samples) / self.sample_rate


class SpeechSegmenter:
    """Bound speech by silence/maximum duration; reset on missing PCM blocks.

    RMS is a simple engineering baseline, not a learned speech detector. A noisy
    room can trigger it; faster-whisper's VAD is applied again during transcription.
    """

    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self._pre: deque[AudioBlock] = deque()
        self._active: list[AudioBlock] = []
        self._voiced = 0.0
        self._silence = 0.0
        self._last_end: float | None = None
        self._identifier = 0
        self.discontinuities = 0

    def push(self, block: AudioBlock) -> list[Utterance]:
        duration = len(block.samples) / self.settings.sample_rate
        gap = self._last_end is not None and abs(block.timestamp_seconds - self._last_end) > 0.1
        if gap or block.discontinuity:
            self.discontinuities += 1
            self._reset()
        self._last_end = block.timestamp_seconds + duration
        rms = float(np.sqrt(np.mean(np.square(block.samples, dtype=np.float64))))
        voiced = rms >= self.settings.rms_threshold
        if not self._active and not voiced:
            self._pre.append(block)
            while self._pre and block.timestamp_seconds + duration - self._pre[0].timestamp_seconds > self.settings.pre_roll_seconds:
                self._pre.popleft()
            return []
        if not self._active:
            self._active = list(self._pre)
            self._pre.clear()
        self._active.append(block)
        self._voiced += duration if voiced else 0.0
        self._silence = 0.0 if voiced else self._silence + duration
        elapsed = self._last_end - self._active[0].timestamp_seconds
        if elapsed >= self.settings.maximum_utterance_seconds:
            result = self._finish("maximum_duration")
        elif self._silence >= self.settings.silence_seconds:
            result = self._finish("silence")
        else:
            result = None
        return [result] if result is not None else []

    def flush(self) -> Utterance | None:
        return self._finish("session_end")

    def _finish(self, reason: str) -> Utterance | None:
        result = None
        if self._active and self._voiced + 1e-8 >= self.settings.minimum_speech_seconds:
            self._identifier += 1
            result = Utterance(
                self._identifier,
                self._active[0].timestamp_seconds,
                np.concatenate([block.samples for block in self._active]).astype(np.float32),
                self.settings.sample_rate,
                reason,
            )
        self._reset()
        return result

    def _reset(self) -> None:
        self._active.clear()
        self._pre.clear()
        self._voiced = self._silence = 0.0


class UtteranceTranscriber:
    """Reuse the existing loaded Whisper model; align local timestamps to session time."""

    def __init__(self, stt_config) -> None:
        self.backend = FasterWhisperTranscriber(stt_config)

    def transcribe(self, utterance: Utterance) -> TranscriptionResult:
        segments, info = self.backend.model.transcribe(
            utterance.samples,
            language=self.backend.config.language,
            vad_filter=self.backend.config.vad_filter,
        )
        aligned = []
        duration = len(utterance.samples) / utterance.sample_rate
        for segment in segments:
            text = segment.text.strip()
            if text:
                start = max(0.0, min(duration, float(segment.start)))
                end = max(start, min(duration, float(segment.end)))
                aligned.append(TranscriptSegment(
                    utterance.start_seconds + start,
                    utterance.start_seconds + end,
                    text,
                ))
        return TranscriptionResult(
            status="success", backend="faster-whisper", model=self.backend.config.model_size,
            language=getattr(info, "language", self.backend.config.language),
            duration_seconds=duration, segments=aligned,
        )
