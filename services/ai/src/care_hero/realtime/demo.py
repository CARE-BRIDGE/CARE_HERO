"""Deterministic fixtures for exercising live orchestration without devices/models.

These results are marked synthetic_demo. They are software verification fixtures,
not detector, speech recognition, or language-model performance measurements.
"""

import threading

import numpy as np

from care_hero.realtime.capture import AudioBlock
from care_hero.schemas import (
    FaceMeshObservation, FrameKeypoints, Keypoint, TranscriptSegment, TranscriptionResult,
)


class DemoCamera:
    def open(self):
        pass

    def read(self):
        return True, np.full((240, 320, 3), 220, dtype=np.uint8)

    def close(self):
        pass


class DemoAnalyzer:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def analyze(self, packet):
        landmarks = {
            "nose": Keypoint(0.5, 0.2, 0.95),
            "left_shoulder": Keypoint(0.35, 0.35, 0.95),
            "right_shoulder": Keypoint(0.65, 0.35, 0.95),
            "left_hip": Keypoint(0.4, 0.65, 0.95),
            "right_hip": Keypoint(0.6, 0.65, 0.95),
        }
        frame = FrameKeypoints(packet.index, packet.timestamp_seconds, True, landmarks, 0.95,
                               FaceMeshObservation(status="not_run"))
        return frame, packet.image, {"pose_ms": 0.0, "face_ms": 0.0}


class DemoMicrophone:
    def __init__(self, settings):
        self.settings = settings
        self.stop = threading.Event()
        self.thread = None

    def start(self, channel, epoch):
        def produce():
            index = 0
            while not self.stop.is_set():
                timestamp = index * self.settings.block_seconds
                # One second of synthetic signal followed by one second of silence.
                amplitude = 0.1 if timestamp % 2 < 1 else 0.0
                count = round(self.settings.sample_rate * self.settings.block_seconds)
                samples = np.full(count, amplitude, dtype=np.float32)
                channel.put(AudioBlock(timestamp, samples))
                index += 1
                self.stop.wait(self.settings.block_seconds)
        self.thread = threading.Thread(target=produce, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)


class DemoTranscriber:
    def transcribe(self, utterance):
        return TranscriptionResult(
            "success", "synthetic_fixture", "fixture-v1", language="ko",
            duration_seconds=utterance.end_seconds - utterance.start_seconds,
            segments=[TranscriptSegment(utterance.start_seconds, utterance.end_seconds,
                                        "많이 속상하시겠어요. 같이 찾아볼게요.")],
        )
