"""Structural interfaces keep hardware/model dependencies replaceable in experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from care_hero.schemas import FrameKeypoints, TranscriptionResult

if TYPE_CHECKING:
    from care_hero.realtime.audio import Utterance
    from care_hero.realtime.capture import AudioBlock, CapturedFrame
    from care_hero.realtime.channels import BoundedChannel


class FrameSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> tuple[bool, Any]: ...
    def close(self) -> None: ...


class FrameAnalyzer(Protocol):
    def __enter__(self) -> Any: ...
    def __exit__(self, *args: Any) -> None: ...
    def analyze(self, packet: CapturedFrame) -> tuple[FrameKeypoints, Any, dict[str, float]]: ...


class AudioSource(Protocol):
    def start(self, channel: BoundedChannel[AudioBlock], epoch: float) -> None: ...
    def close(self) -> None: ...


class TurnTranscriber(Protocol):
    def transcribe(self, utterance: Utterance) -> TranscriptionResult: ...
