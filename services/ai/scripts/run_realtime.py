#!/usr/bin/env python3
"""Local live camera/microphone runner and reproducible hardware-free demo."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from care_hero.config import load_config  # noqa: E402
from care_hero.realtime.audio import UtteranceTranscriber  # noqa: E402
from care_hero.realtime.capture import Microphone, OpenCVCamera  # noqa: E402
from care_hero.realtime.language import PromptBundle, StructuredLanguageModel  # noqa: E402
from care_hero.realtime.session import RealtimeSession  # noqa: E402
from care_hero.realtime.settings import load_realtime_settings  # noqa: E402
from care_hero.realtime.vision import VisionAnalyzer  # noqa: E402
from care_hero.scoring import load_scenario_rules  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--audio", action="store_true", help="Enable microphone and utterance STT")
    parser.add_argument("--audio-only", action="store_true", help="Microphone input without camera")
    parser.add_argument("--audio-device", type=int)
    parser.add_argument("--llm", action="store_true", help="Explicitly enable external text-only LLM requests")
    parser.add_argument("--model", help="Structured-output-compatible OpenAI model; or OPENAI_MODEL")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--demo", action="store_true", help="Synthetic camera, audio and STT; no hardware/API")
    parser.add_argument("--duration", type=float, help="Capture duration in seconds (maximum 600)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/realtime.yaml")
    parser.add_argument("--base-config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--scenario", type=Path, default=PROJECT_ROOT / "configs/scenarios/dementia_first_visit.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.demo and args.llm:
        parser.error("--demo is API-free; use mocked LLM tests to verify the adapter.")
    if args.demo and args.audio_only:
        parser.error("--demo includes both synthetic video and audio.")
    if args.llm and (not os.getenv("OPENAI_API_KEY") or not (args.model or os.getenv("OPENAI_MODEL"))):
        parser.error("--llm requires OPENAI_API_KEY and --model or OPENAI_MODEL.")
    settings = load_realtime_settings(args.config)
    if args.duration is not None:
        settings = type(settings).model_validate({**settings.model_dump(), "duration_seconds": args.duration})
    elif args.demo:
        settings = type(settings).model_validate({**settings.model_dump(), "duration_seconds": 4})
    config, rules = load_config(args.base_config), load_scenario_rules(args.scenario)
    language = None
    if args.llm:
        language = StructuredLanguageModel(PromptBundle(PROJECT_ROOT / "prompts/realtime"), settings.llm, model=args.model)
    output = args.output or REPOSITORY_ROOT / "outputs/realtime" / f"{datetime.now().astimezone():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    if args.demo:
        from care_hero.realtime.demo import DemoAnalyzer, DemoCamera, DemoMicrophone, DemoTranscriber
        source, analyzer = DemoCamera(), DemoAnalyzer()
        microphone, transcriber_factory = DemoMicrophone(settings.audio), DemoTranscriber
    else:
        source = None if args.audio_only else OpenCVCamera(args.camera)
        analyzer = nullcontext() if args.audio_only else VisionAnalyzer(config)
        microphone = Microphone(settings.audio, args.audio_device) if args.audio or args.audio_only else None
        transcriber_factory = lambda: UtteranceTranscriber(config.stt)
    session = RealtimeSession(
        settings, rules, source=source, analyzer=analyzer, output=output,
        microphone=microphone, transcriber_factory=transcriber_factory,
        language_model=language, preview=not args.headless and not args.demo and not args.audio_only,
        synthetic=args.demo,
    )
    result = session.run()
    print(f"Session status: {result['status']}; result: {output / 'session.json'}")
    return 0 if result["status"] in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
