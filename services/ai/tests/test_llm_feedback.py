from __future__ import annotations

from types import SimpleNamespace

import pytest

from care_hero.errors import LLMConfigurationError
from care_hero.llm_feedback import FeedbackItem, LLMFeedback, PracticeStep, generate_llm_feedback
from care_hero.schemas import RuleBasedReport


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def test_llm_feedback_uses_fixed_pydantic_schema() -> None:
    parsed = LLMFeedback(
        schema_version="1.0",
        overall_summary="규칙 결과를 바탕으로 자세를 점검했습니다.",
        strengths=[
            FeedbackItem(
                metric_id="front_facing_ratio",
                title="정면 자세",
                message="정면 자세가 안정적입니다.",
                evidence="front_facing_ratio=0.95",
                action="같은 자세를 유지하세요.",
            )
        ],
        improvements=[],
        practice_plan=[
            PracticeStep(order=1, instruction="정면을 향해 인사하세요.", success_check="어깨 중심을 확인하세요.")
        ],
        caution="provisional 기준이며 의료 판단이 아닙니다.",
    )
    client = FakeClient(parsed)
    report = RuleBasedReport("sample", "Sample", "provisional", "valid", 90.0, [])
    result = generate_llm_feedback(report, client=client, model="test-model")
    assert result is parsed
    assert client.responses.request["model"] == "test-model"
    assert client.responses.request["text_format"] is LLMFeedback
    assert client.responses.request["store"] is False
    assert "frames" not in client.responses.request["input"][1]["content"]


def test_llm_feedback_requires_api_key_without_injected_client(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = RuleBasedReport("sample", "Sample", "provisional", "valid", 90.0, [])
    with pytest.raises(LLMConfigurationError):
        generate_llm_feedback(report)
