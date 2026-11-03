"""Validated controls for sampling, backpressure, endpointing and shutdown."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_rate: int = Field(default=16000, ge=16000, le=16000)
    block_seconds: float = Field(default=0.02, ge=0.01, le=0.1)
    rms_threshold: float = Field(default=0.012, gt=0, lt=1)
    minimum_speech_seconds: float = Field(default=0.4, gt=0)
    silence_seconds: float = Field(default=0.8, gt=0)
    maximum_utterance_seconds: float = Field(default=15, gt=0, le=60)
    pre_roll_seconds: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_lengths(self):
        if self.maximum_utterance_seconds <= self.minimum_speech_seconds + self.pre_roll_seconds:
            raise ValueError("Maximum utterance must exceed speech minimum plus pre-roll.")
        return self


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout_seconds: float = Field(default=15, gt=0, le=120)
    history_turns: int = Field(default=4, ge=0, le=10)
    maximum_transcript_characters: int = Field(default=3000, ge=100, le=10000)


class RealtimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_seconds: float = Field(default=180, gt=0, le=600)
    analysis_hz: float = Field(default=10, ge=1, le=30)
    window_seconds: float = Field(default=5, ge=1, le=30)
    snapshot_interval_seconds: float = Field(default=1, ge=0.1, le=10)
    video_queue_size: int = Field(default=2, ge=1, le=8)
    audio_queue_size: int = Field(default=64, ge=2, le=512)
    utterance_queue_size: int = Field(default=4, ge=1, le=16)
    llm_queue_size: int = Field(default=4, ge=1, le=16)
    drain_timeout_seconds: float = Field(default=30, gt=0, le=120)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


def load_realtime_settings(path: str | Path) -> RealtimeSettings:
    return RealtimeSettings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
