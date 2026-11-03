"""Versioned prompt templates + typed outputs + evidence checks.

No numeric score is returned by the language model. Locally computed rule results
remain authoritative and are copied into the final report by the session runner.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from care_hero.llm_feedback import LLMFeedback


class TurnInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    response_type: Literal["acknowledgment", "reassurance", "question", "denial", "other", "uncertain"]
    evidence_quote: str = Field(max_length=250)
    observation: str = Field(min_length=1, max_length=400)
    uncertainty: str = Field(min_length=1, max_length=250)


class PromptBundle:
    def __init__(self, root: Path) -> None:
        self.text = {
            name: (root / name).read_text(encoding="utf-8")
            for name in ("turn_system.txt", "turn_user.template.txt", "final_system.txt", "final_user.template.txt")
        }

    def messages(self, task: str, evidence: dict) -> list[dict]:
        return [
            {"role": "developer", "content": self.text[f"{task}_system.txt"]},
            {"role": "user", "content": Template(self.text[f"{task}_user.template.txt"]).substitute(
                evidence_json=json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
            )},
        ]

    def manifest(self) -> dict:
        return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in self.text.items()}


class StructuredLanguageModel:
    def __init__(self, prompts: PromptBundle, settings, *, model: str | None = None, client=None) -> None:
        self.prompts = prompts
        self.settings = settings
        self.model = model or os.getenv("OPENAI_MODEL")
        if not self.model:
            raise ValueError("Set OPENAI_MODEL explicitly for structured output support.")
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=settings.timeout_seconds, max_retries=0)
        self.client = client

    def _request(self, task: str, evidence: dict, schema):
        response = self.client.responses.parse(
            model=self.model,
            store=False,
            max_output_tokens=2000,
            input=self.prompts.messages(task, evidence),
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("No structured output: refusal or incomplete generation.")
        return schema.model_validate(parsed)

    def interpret_turn(self, transcript: str, window: dict, history: list[dict]) -> TurnInterpretation:
        text = transcript[:self.settings.maximum_transcript_characters]
        context = history[-self.settings.history_turns:] if self.settings.history_turns else []
        result = self._request("turn", {
            "transcript": text, "vision_window": window, "recent_turns": context,
            "scope": "spoken text only; no voice emotion or facial emotion evidence",
        }, TurnInterpretation)
        if result.evidence_quote and result.evidence_quote not in text:
            raise ValueError("LLM evidence is not an exact transcript substring.")
        if result.response_type != "uncertain" and not result.evidence_quote:
            raise ValueError("A non-uncertain interpretation requires transcript evidence.")
        return result

    def final_feedback(self, rules, turns: list[dict]) -> LLMFeedback:
        result = self._request("final", {
            "rule_based": rules.to_dict(),
            "recent_turns": turns[-self.settings.history_turns:] if self.settings.history_turns else [],
        }, LLMFeedback)
        available = {metric.metric_id: metric for metric in rules.metrics}
        for item in result.strengths + result.improvements:
            if item.metric_id not in available:
                raise ValueError("Unknown metric ID in model response.")
        for item in result.strengths:
            if available[item.metric_id].status != "valid":
                raise ValueError("Unavailable measurements cannot support a strength.")
        return result
