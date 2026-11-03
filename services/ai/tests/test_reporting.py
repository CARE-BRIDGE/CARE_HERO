from __future__ import annotations

import json

from care_hero.reporting import build_final_report, save_report_json, save_report_markdown
from care_hero.schemas import (
    RuleBasedReport,
    TranscriptSegment,
    TranscriptionResult,
    VideoKeypointResult,
    VideoMetadata,
)


def test_reports_are_saved_without_llm_feedback(tmp_path) -> None:
    keypoints = VideoKeypointResult(VideoMetadata(30.0, 0, 0.0, 640, 360), [])
    rules = RuleBasedReport(
        "sample",
        "Sample",
        "provisional",
        "insufficient_data",
        None,
        [],
        ["공식 기준이 아닙니다."],
    )
    report = build_final_report(
        keypoints,
        rules,
        llm_status="error",
        llm_feedback=None,
        llm_error="API key missing",
    )
    json_path = save_report_json(report, tmp_path / "report.json")
    markdown_path = save_report_markdown(report, tmp_path / "report.md")
    assert json.loads(json_path.read_text(encoding="utf-8"))["rule_based"]["total_score"] is None
    assert "측정 불가" in markdown_path.read_text(encoding="utf-8")


def test_report_includes_optional_transcription(tmp_path) -> None:
    keypoints = VideoKeypointResult(VideoMetadata(30.0, 0, 0.0, 640, 360), [])
    rules = RuleBasedReport("sample", "Sample", "provisional", "partial", None, [])
    transcription = TranscriptionResult(
        status="success",
        backend="fake",
        model="test",
        language="ko",
        segments=[TranscriptSegment(0.0, 1.0, "반갑습니다")],
    )
    report = build_final_report(
        keypoints,
        rules,
        llm_status="skipped",
        llm_feedback=None,
        transcription=transcription,
    )
    markdown_path = save_report_markdown(report, tmp_path / "report.md")
    assert "반갑습니다" in markdown_path.read_text(encoding="utf-8")
