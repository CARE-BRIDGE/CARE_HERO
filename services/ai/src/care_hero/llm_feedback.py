"""OpenAI Responses API adapter with a strict feedback output contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from care_hero.errors import LLMConfigurationError, LLMResponseError
from care_hero.schemas import RuleBasedReport


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "feedback_system.txt"
DEFAULT_MODEL = "gpt-5.6-sol"


class FeedbackItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=400)
    evidence: str = Field(min_length=1, max_length=300)
    action: str = Field(min_length=1, max_length=300)


class PracticeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1, le=3)
    instruction: str = Field(min_length=1, max_length=300)
    success_check: str = Field(min_length=1, max_length=250)


class LLMFeedback(BaseModel):
    """Fixed response template returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    overall_summary: str = Field(min_length=1, max_length=500)
    strengths: list[FeedbackItem] = Field(max_length=3)
    improvements: list[FeedbackItem] = Field(max_length=3)
    practice_plan: list[PracticeStep] = Field(min_length=1, max_length=3)
    caution: str = Field(min_length=1, max_length=300)


def generate_llm_feedback(
    rule_report: RuleBasedReport,
    *,
    client: Any | None = None,
    model: str | None = None,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
) -> LLMFeedback:
    """Generate Korean feedback without allowing the model to alter rule results."""

    if client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMConfigurationError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        try:
            from openai import OpenAI
        except (ImportError, OSError) as exc:
            raise LLMConfigurationError("openai Python 패키지를 불러올 수 없습니다.") from exc
        client = OpenAI()
    system_prompt = _load_prompt(prompt_path)
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    compact_input = json.dumps(rule_report.to_dict(), ensure_ascii=False, separators=(",", ":"))
    try:
        response = client.responses.parse(
            model=selected_model,
            reasoning={"effort": "low"},
            store=False,
            input=[
                {"role": "developer", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "다음 규칙 기반 결과만 근거로 피드백을 작성하세요. "
                        "점수와 측정값을 변경하지 마세요.\n" + compact_input
                    ),
                },
            ],
            text_format=LLMFeedback,
        )
    except Exception as exc:
        raise LLMResponseError("LLM 피드백 생성 요청에 실패했습니다.") from exc
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise LLMResponseError("LLM이 구조화된 피드백을 반환하지 않았습니다.")
    if isinstance(parsed, LLMFeedback):
        return parsed
    try:
        return LLMFeedback.model_validate(parsed)
    except Exception as exc:
        raise LLMResponseError("LLM 피드백 스키마 검증에 실패했습니다.") from exc


def _load_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise LLMConfigurationError("LLM 피드백 prompt 파일을 찾을 수 없습니다.")
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMConfigurationError("LLM 피드백 prompt를 읽을 수 없습니다.") from exc
