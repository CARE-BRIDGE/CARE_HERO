"""Hardware adapters: capture only, never run model inference in a callback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from care_hero.realtime.channels import BoundedChannel
from care_hero.realtime.settings import AudioSettings


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    timestamp_seconds: float
    image: Any


@dataclass(frozen=True)
class AudioBlock:
    timestamp_seconds: float
    samples: np.ndarray
    discontinuity: bool = False


class OpenCVCamera:
    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self.capture: Any = None

    def open(self) -> None:
        import cv2

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError("camera_open_failed")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


class Microphone:
    """Copy short PCM blocks and return immediately to PortAudio."""

    def __init__(self, settings: AudioSettings, device: int | None = None) -> None:
        self.settings = settings
        self.device = device
        self.stream: Any = None

    def start(self, channel: BoundedChannel[AudioBlock], epoch: float) -> None:
        import sounddevice as sd

        def callback(indata, frames, timing, status):
            # Convert PortAudio's clock to our shared monotonic session clock.
            delay = max(0.0, timing.currentTime - timing.inputBufferAdcTime)
            timestamp = max(0.0, time.monotonic() - epoch - delay)
            channel.put(AudioBlock(timestamp, indata[:, 0].copy(), bool(status)))

        self.stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            blocksize=round(self.settings.sample_rate * self.settings.block_seconds),
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self.stream.start()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
