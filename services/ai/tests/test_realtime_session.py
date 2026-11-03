import json
import math
from pathlib import Path

import pytest

from care_hero.realtime.demo import DemoAnalyzer, DemoCamera, DemoMicrophone, DemoTranscriber
from care_hero.realtime.metrics import OnlineMetrics
from care_hero.realtime.session import RealtimeSession
from care_hero.realtime.settings import AudioSettings, RealtimeSettings
from care_hero.realtime.audio import Utterance
import numpy as np
from care_hero.scoring import load_scenario_rules
from care_hero.schemas import FrameKeypoints, Keypoint

ROOT = Path(__file__).resolve().parents[1]


def rules():
    return load_scenario_rules(ROOT / "configs/scenarios/dementia_first_visit.yaml")


def test_metrics_correct_non_square_image_angles_and_expire_window():
    settings = RealtimeSettings(duration_seconds=10, window_seconds=1)
    metrics = OnlineMetrics(settings, rules())
    for index in range(25):
        points = {
            "nose": Keypoint(0.6, 0.1, 1),
            "left_shoulder": Keypoint(0.5, 0.3, 1),
            "right_shoulder": Keypoint(0.7, 0.3, 1),
            "left_hip": Keypoint(0.4, 0.7, 1),
            "right_hip": Keypoint(0.6, 0.7, 1),
        }
        metrics.update(FrameKeypoints(index, index / 10, True, points), 800, 400)
    window = metrics.snapshot()
    assert window["analyzed_frame_count"] <= 11
    tilt = next(item for item in window["rule_based"]["metrics"] if item["metric_id"] == "torso_alignment")
    assert tilt["raw_value"] == pytest.approx(math.degrees(math.atan2(0.2, 0.4)))


def test_synthetic_realtime_session_generates_synchronized_outputs(tmp_path):
    settings = RealtimeSettings(
        duration_seconds=0.7, analysis_hz=15, snapshot_interval_seconds=0.1,
        audio=AudioSettings(minimum_speech_seconds=0.1),
    )
    result = RealtimeSession(
        settings, rules(), source=DemoCamera(), analyzer=DemoAnalyzer(), output=tmp_path,
        microphone=DemoMicrophone(settings.audio), transcriber_factory=DemoTranscriber,
        synthetic=True,
    ).run()
    assert result["status"] == "completed"
    assert result["mode"] == "synthetic_demo"
    assert result["summary"]["analyzed_frames"] > 0
    assert result["turns"][0]["transcription"]["backend"] == "synthetic_fixture"
    assert result["unfinished_workers"] == []
    saved = json.loads((tmp_path / "session.json").read_text())
    assert saved["capture_duration_seconds"] > 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [item["sequence"] for item in events] == list(range(1, len(events) + 1))
    assert all("landmarks" not in json.dumps(item) for item in events)
    assert (tmp_path / "report.md").is_file()


def test_failed_camera_produces_insufficient_data_report_and_cleans_up(tmp_path):
    class BrokenCamera(DemoCamera):
        closed = False
        def open(self):
            raise RuntimeError("private device details")
        def close(self):
            self.closed = True
    source = BrokenCamera()
    result = RealtimeSession(RealtimeSettings(duration_seconds=0.2), rules(), source=source,
                             analyzer=DemoAnalyzer(), output=tmp_path).run()
    assert source.closed
    assert result["status"] == "partial"
    assert result["rule_based"]["total_score"] is None
    assert "private device details" not in (tmp_path / "events.jsonl").read_text()


def test_session_never_overwrites_existing_run(tmp_path):
    (tmp_path / "events.jsonl").write_text("existing")
    with pytest.raises(FileExistsError):
        RealtimeSession(RealtimeSettings(), rules(), source=DemoCamera(), analyzer=DemoAnalyzer(), output=tmp_path)
    assert (tmp_path / "events.jsonl").read_text() == "existing"


def test_stt_failure_does_not_stop_visual_measurements(tmp_path):
    class FailingTranscriber:
        def transcribe(self, utterance):
            raise RuntimeError("private transcript or path")
    settings = RealtimeSettings(duration_seconds=0.3, audio=AudioSettings(minimum_speech_seconds=0.05))
    result = RealtimeSession(
        settings, rules(), source=DemoCamera(), analyzer=DemoAnalyzer(), output=tmp_path,
        microphone=DemoMicrophone(settings.audio), transcriber_factory=FailingTranscriber,
    ).run()
    assert result["summary"]["analyzed_frames"] > 0
    assert result["turns"][0]["stt_status"] == "error"
    assert result["turns"][0]["error_code"] == "RuntimeError"
    assert "private transcript" not in json.dumps(result)


def test_delayed_endpoint_never_uses_a_future_visual_window(tmp_path):
    session = RealtimeSession(RealtimeSettings(), rules(), source=DemoCamera(),
                              analyzer=DemoAnalyzer(), output=tmp_path)
    session._snapshots.extend([
        {"window_end_seconds": 1.0, "marker": "past"},
        {"window_end_seconds": 3.0, "marker": "future"},
    ])
    session._submit_utterance(Utterance(1, 1.0, np.zeros(16000, dtype=np.float32), 16000, "silence"))
    _, context = session.speech.get()
    assert context["marker"] == "past"
    session.events.close()


def test_language_worker_and_final_schema_run_without_an_external_api(tmp_path):
    from care_hero.realtime.language import TurnInterpretation
    from care_hero.llm_feedback import LLMFeedback, PracticeStep
    from types import SimpleNamespace

    class FakeLanguage:
        model = "mock-only"
        prompts = SimpleNamespace(manifest=lambda: {"fixture": "not-a-production-prompt"})
        def interpret_turn(self, transcript, window, history):
            return TurnInterpretation(schema_version="1.0", response_type="acknowledgment",
                                      evidence_quote="많이 속상하시겠어요", observation="상대의 표현을 언급했습니다.",
                                      uncertainty="합성 시험값입니다.")
        def final_feedback(self, report, turns):
            return LLMFeedback(schema_version="1.0", overall_summary="테스트 피드백",
                               strengths=[], improvements=[],
                               practice_plan=[PracticeStep(order=1, instruction="재촬영", success_check="좌표 확인")],
                               caution="합성 통합 시험")
    settings = RealtimeSettings(duration_seconds=0.3, audio=AudioSettings(minimum_speech_seconds=0.05))
    result = RealtimeSession(settings, rules(), source=DemoCamera(), analyzer=DemoAnalyzer(), output=tmp_path,
                             microphone=DemoMicrophone(settings.audio), transcriber_factory=DemoTranscriber,
                             language_model=FakeLanguage(), synthetic=True).run()
    assert result["llm_status"] == "success"
    assert result["turns"][0]["llm_status"] == "success"
    assert result["llm_manifest"]["model"] == "mock-only"
    assert result["rule_based"]["total_score"] is None
