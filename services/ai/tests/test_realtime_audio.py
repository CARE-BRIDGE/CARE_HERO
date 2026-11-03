import numpy as np
import pytest

from care_hero.realtime.audio import SpeechSegmenter, Utterance, UtteranceTranscriber
from care_hero.realtime.capture import AudioBlock
from care_hero.realtime.channels import BoundedChannel
from care_hero.realtime.settings import AudioSettings


def block(timestamp, amplitude):
    return AudioBlock(timestamp, np.full(320, amplitude, dtype=np.float32))


def test_video_channel_replaces_old_frames_but_work_channel_rejects_new():
    latest = BoundedChannel(2, replace_oldest=True)
    for index in range(3):
        latest.put(index)
    assert latest.dropped == 1
    assert [latest.get(), latest.get()] == [1, 2]
    work = BoundedChannel(1, replace_oldest=False)
    assert work.put("first")
    assert not work.put("second")
    assert work.get() == "first"
    assert work.dropped == 1


def test_segmenter_ends_on_silence_and_retains_session_offset():
    segmenter = SpeechSegmenter(AudioSettings(silence_seconds=0.1, minimum_speech_seconds=0.1))
    output = []
    for index in range(12):
        output.extend(segmenter.push(block(3 + index * 0.02, 0.1 if index < 6 else 0)))
    assert len(output) == 1
    assert output[0].start_seconds == pytest.approx(3)
    assert output[0].reason == "silence"
    assert segmenter.flush() is None


def test_segmenter_drops_subminimum_noise_and_flushes_unfinished_speech():
    segmenter = SpeechSegmenter(AudioSettings(minimum_speech_seconds=0.1))
    segmenter.push(block(0, 0.1))
    assert segmenter.flush() is None
    for index in range(6):
        segmenter.push(block(1 + index * 0.02, 0.1))
    utterance = segmenter.flush()
    assert utterance.reason == "session_end"
    assert utterance.start_seconds == pytest.approx(1)


def test_audio_gap_never_stitches_separated_audio_into_one_utterance():
    segmenter = SpeechSegmenter(AudioSettings(minimum_speech_seconds=0.1))
    for index in range(6):
        segmenter.push(block(index * 0.02, 0.1))
    segmenter.push(block(2, 0.1))
    assert segmenter.discontinuities == 1
    assert segmenter.flush() is None


def test_segmenter_bounds_a_continuous_utterance():
    settings = AudioSettings(minimum_speech_seconds=0.05, pre_roll_seconds=0, maximum_utterance_seconds=0.2)
    segmenter = SpeechSegmenter(settings)
    output = []
    for index in range(21):
        output.extend(segmenter.push(block(index * 0.02, 0.1)))
    assert len(output) >= 2
    assert all(item.end_seconds - item.start_seconds <= 0.221 for item in output)


def test_whisper_segment_times_are_clamped_and_aligned_without_loading_weights():
    from types import SimpleNamespace

    transcriber = object.__new__(UtteranceTranscriber)
    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
        iter([SimpleNamespace(start=-0.1, end=5, text=" 테스트 ")]), SimpleNamespace(language="ko")
    ))
    transcriber.backend = SimpleNamespace(
        model=model, config=SimpleNamespace(language="ko", vad_filter=True, model_size="fake")
    )
    result = transcriber.transcribe(Utterance(1, 10, np.zeros(16000, dtype=np.float32), 16000, "silence"))
    assert result.segments[0].start_seconds == 10
    assert result.segments[0].end_seconds == 11
