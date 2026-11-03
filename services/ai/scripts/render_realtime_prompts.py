#!/usr/bin/env python3
"""Render reviewable LLM requests and JSON schemas offline (no API key required)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from care_hero.realtime.language import PromptBundle, TurnInterpretation  # noqa: E402
from care_hero.llm_feedback import LLMFeedback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs/prompt_preview")
    args = parser.parse_args()
    bundle = PromptBundle(ROOT / "prompts/realtime")
    payload = {
        "mode": "synthetic_prompt_preview",
        "messages": bundle.messages("turn", {
            "transcript": "많이 속상하시겠어요. 제가 같이 찾아볼게요.",
            "vision_window": {"window_start_seconds": 0, "window_end_seconds": 5,
                              "rule_based": {"overall_status": "insufficient_data", "total_score": None}},
            "recent_turns": [],
        }),
        "turn_output_schema": TurnInterpretation.model_json_schema(),
        "final_output_schema": LLMFeedback.model_json_schema(),
        "prompts_sha256": bundle.manifest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request_preview.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Offline prompt/schema preview: {args.output / 'request_preview.json'}")


if __name__ == "__main__":
    main()
