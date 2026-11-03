"""Final JSON and Markdown report assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from care_hero.llm_feedback import LLMFeedback
from care_hero.schemas import (
    FinalAnalysisReport,
    RuleBasedReport,
    TranscriptionResult,
    VideoKeypointResult,
)


def build_final_report(
    keypoints: VideoKeypointResult,
    rules: RuleBasedReport,
    *,
    llm_status: str,
    llm_feedback: LLMFeedback | None,
    llm_error: str | None = None,
    transcription: TranscriptionResult | None = None,
) -> FinalAnalysisReport:
    summary = keypoints.to_dict()["summary"]
    return FinalAnalysisReport(
        schema_version="1.1",
        generated_at=datetime.now(UTC).isoformat(),
        video_summary=summary,
        rule_based=rules,
        llm_status=llm_status,
        llm_feedback=llm_feedback.model_dump() if llm_feedback is not None else None,
        llm_error=llm_error,
        transcription=transcription,
    )


def save_report_json(report: FinalAnalysisReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return path


def save_report_markdown(report: FinalAnalysisReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rule = report.rule_based
    score_text = f"{rule.total_score:.2f}/100" if rule.total_score is not None else "측정 불가"
    face_summary = report.video_summary.get("face_mesh", {})
    face_analyzed = int(face_summary.get("analyzed_frame_count") or 0)
    face_detected = int(face_summary.get("detected_frame_count") or 0)
    face_text = (
        f"{face_detected}/{face_analyzed} 프레임"
        if face_analyzed
        else "사용 안 함"
    )
    lines = [
        "# CARE:HERO 행동 리허설 결과",
        "",
        f"- 시나리오: {rule.scenario_name}",
        f"- 규칙 평가 상태: `{rule.overall_status}`",
        f"- provisional 총점: **{score_text}**",
        f"- FaceMesh 검출: **{face_text}**",
        f"- LLM 상태: `{report.llm_status}`",
        "",
        "## 규칙 기반 결과",
        "",
        "| 항목 | 원시값 | 상태 | 점수 | 피드백 |",
        "|---|---:|---|---:|---|",
    ]
    for metric in rule.metrics:
        raw = "-" if metric.raw_value is None else f"{metric.raw_value:.4f} {metric.unit}"
        score = "-" if metric.score is None else f"{metric.score:.2f}"
        lines.append(
            f"| {metric.display_name} | {raw} | {metric.status} | {score} | {metric.feedback} |"
        )
    lines.extend(["", "## 음성 전사", ""])
    if report.transcription is None:
        lines.append("STT를 사용하지 않았습니다.")
    elif report.transcription.status != "success":
        lines.append(report.transcription.error or "음성 전사를 완료하지 못했습니다.")
    else:
        language = report.transcription.language or "unknown"
        lines.extend(
            [
                f"- 언어: `{language}`",
                f"- 모델: `{report.transcription.backend}/{report.transcription.model}`",
                "",
                report.transcription.full_text or "전사된 발화가 없습니다.",
            ]
        )
    lines.extend(["", "## LLM 피드백", ""])
    if report.llm_feedback is None:
        lines.append(report.llm_error or "LLM 피드백이 생성되지 않았습니다.")
    else:
        feedback = report.llm_feedback
        lines.extend([feedback["overall_summary"], "", "### 장점", ""])
        lines.extend(
            f"- **{item['title']}**: {item['message']} ({item['evidence']})"
            for item in feedback["strengths"]
        )
        lines.extend(["", "### 개선점", ""])
        lines.extend(
            f"- **{item['title']}**: {item['message']} 다음 행동: {item['action']}"
            for item in feedback["improvements"]
        )
        lines.extend(["", "### 다음 연습", ""])
        lines.extend(
            f"{step['order']}. {step['instruction']} — 확인: {step['success_check']}"
            for step in feedback["practice_plan"]
        )
        lines.extend(["", f"> {feedback['caution']}"])
    lines.extend(["", "## 측정 제외 및 주의사항", ""])
    lines.extend(f"- {caveat}" for caveat in rule.caveats)
    _atomic_write(path, "\n".join(lines) + "\n")
    return path


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
