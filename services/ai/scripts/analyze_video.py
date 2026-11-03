#!/usr/bin/env python3
"""Local Pose, FaceMesh, rule, STT, and structured feedback pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from care_hero.config import load_config  # noqa: E402
from care_hero.errors import (  # noqa: E402
    CareHeroError,
    LLMConfigurationError,
    LLMResponseError,
    SpeechTranscriptionError,
)
from care_hero.llm_feedback import generate_llm_feedback  # noqa: E402
from care_hero.pipeline import extract_video_keypoints, save_keypoints_json  # noqa: E402
from care_hero.reporting import build_final_report, save_report_json, save_report_markdown  # noqa: E402
from care_hero.scoring import load_scenario_rules, score_keypoints  # noqa: E402
from care_hero.schemas import TranscriptionResult  # noqa: E402
from care_hero.stt import save_transcription_json, transcribe_media  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract keypoints, score provisional rules, and generate structured feedback."
    )
    parser.add_argument("--video", required=True, help="Input mp4, mov, or avi path")
    parser.add_argument(
        "--scenario",
        default=str(PROJECT_ROOT / "configs" / "scenarios" / "dementia_first_visit.yaml"),
    )
    parser.add_argument("--keypoints-output", default="outputs/keypoints.json")
    parser.add_argument("--report-json", default="outputs/report.json")
    parser.add_argument("--report-md", default="outputs/report.md")
    parser.add_argument("--stt", action="store_true", help="Run optional local faster-whisper STT")
    parser.add_argument("--transcript-output", default="outputs/transcript.json")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Create only the rule report without calling OpenAI",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_config()

    def show_progress(current: int, total: int) -> None:
        print(f"\rkeypoint extraction: {current}/{total}", end="", flush=True)

    try:
        keypoints = extract_video_keypoints(
            args.video,
            config=settings,
            progress_callback=show_progress,
        )
        save_keypoints_json(keypoints, args.keypoints_output)
        rule_report = score_keypoints(keypoints, load_scenario_rules(args.scenario))
    except CareHeroError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    transcription = None
    if args.stt or settings.stt.enabled:
        try:
            transcription = transcribe_media(args.video, settings.stt)
        except SpeechTranscriptionError as exc:
            transcription = TranscriptionResult(
                status="error",
                backend="faster-whisper",
                model=settings.stt.model_size,
                error=str(exc),
            )
            exit_code = 3
        save_transcription_json(transcription, args.transcript_output)

    llm_feedback = None
    llm_status = "skipped"
    llm_error = None
    if not args.skip_llm:
        try:
            llm_feedback = generate_llm_feedback(rule_report)
            llm_status = "success"
        except (LLMConfigurationError, LLMResponseError) as exc:
            llm_status = "error"
            llm_error = str(exc)
            exit_code = max(exit_code, 2)

    report = build_final_report(
        keypoints,
        rule_report,
        llm_status=llm_status,
        llm_feedback=llm_feedback,
        llm_error=llm_error,
        transcription=transcription,
    )
    save_report_json(report, args.report_json)
    save_report_markdown(report, args.report_md)
    print()
    print(f"rule status: {rule_report.overall_status}")
    print(
        "provisional score:",
        f"{rule_report.total_score:.2f}" if rule_report.total_score is not None else "unavailable",
    )
    print(f"llm status: {llm_status}")
    print(f"stt status: {transcription.status if transcription is not None else 'skipped'}")
    print(f"keypoints: {Path(args.keypoints_output).resolve()}")
    print(f"report json: {Path(args.report_json).resolve()}")
    print(f"report markdown: {Path(args.report_md).resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
