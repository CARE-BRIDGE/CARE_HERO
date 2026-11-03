# CARE:HERO LLM engineering: templates, contracts, evidence

## 설계 목적

실시간 경로의 LLM은 완료된 발화의 표현을 관찰하고, 종료 후 로컬 측정 결과를 이해하기 쉬운 문장으로 설명한다. 입력은 텍스트와 작은 규칙 평가 요약이며, 영상·음성 원본이나 전체 landmark 배열은 전송하지 않는다.

구현은 [`language.py`](../services/ai/src/care_hero/realtime/language.py)의 `PromptBundle`과 `StructuredLanguageModel`에 있다. [OpenAI Structured Outputs 공식 문서](https://developers.openai.com/api/docs/guides/structured-outputs)에 따른 `responses.parse(..., text_format=PydanticModel)`을 사용한다. `.output_parsed`가 없거나 스키마 검증에 실패하면 오류로 기록한다.

## 요청 단위

| 요청 | 호출 시점 | 입력 근거 | 출력 |
|---|---|---|---|
| `interpret_turn()` | 유효 발화의 STT 완료 후 | 잘린 길이의 전사, 발화 확정 시의 자세 윈도우, 최근 최대 4개 해석 | 발화 유형·정확한 인용·관찰·불확실성 |
| `final_feedback()` | 입력 종료와 작업 큐 정리 후 | 전체 세션 규칙 평가, 최근 최대 4개 발화 해석 | 요약·장점·개선점·연습 계획 |

일정한 시간마다 무조건 호출하거나 프레임마다 호출하지 않는다. STT가 비어 있으면 해당 발화의 LLM 호출을 생략한다. 전체 대화 분석이 필요한 경우에는 현재의 bounded history를 별도 요약 메모리로 확장해야 한다.

## 템플릿 파일

| 파일 | 책임 |
|---|---|
| [`turn_system.txt`](../services/ai/prompts/realtime/turn_system.txt) | 역할, 발화 분류 정의, 인용 규칙, 근거 한계, 중간 코칭 금지 |
| [`turn_user.template.txt`](../services/ai/prompts/realtime/turn_user.template.txt) | 발화와 자세 요약을 JSON 데이터로 주입 |
| [`final_system.txt`](../services/ai/prompts/realtime/final_system.txt) | 로컬 점수 보존, 유효 지표만 장점 근거로 사용, 누락값 설명 |
| [`final_user.template.txt`](../services/ai/prompts/realtime/final_user.template.txt) | 세션 평가와 최근 발화 해석을 JSON 데이터로 주입 |

개발자 메시지는 역할과 계약을 정의하고, 사용자 메시지는 관찰 데이터를 담는다. Python `string.Template`의 `$evidence_json` 한 곳만 치환하며, 사용자의 실제 발화는 JSON 문자열 값으로 직렬화한다. 발화 안에 명령문이 있어도 데이터로 취급하도록 상위 지침에 명시한다. 이러한 분리는 공격 저항성을 높이지만 모든 프롬프트 공격을 완벽히 차단한다는 보장은 아니다.

## 발화 출력 스키마

```json
{
  "schema_version": "1.0",
  "response_type": "acknowledgment",
  "evidence_quote": "많이 속상하시겠어요",
  "observation": "상대가 표현한 감정을 언급하는 말이 포함되었습니다.",
  "uncertainty": "발화 문구만으로 실제 공감의 정도는 판단하지 않습니다."
}
```

예시는 출력 형식 설명용이며 실제 모델 호출 결과가 아니다. `response_type`은 다음 중 하나다.

- `acknowledgment`: 상대의 말·감정을 언급하거나 인정하는 표현
- `reassurance`: 안심시키거나 함께 돕겠다는 표현
- `question`: 질문 중심 응답
- `denial`: 사실을 부정하는 표현
- `other`: 위 범주에 맞지 않는 관찰 가능한 발화
- `uncertain`: 짧음·모호함·불충분한 근거로 판단 유보

하나의 문장에 여러 표현이 있으면 현재 스키마는 하나의 대표 유형만 반환한다. 이 분류는 교육용 관찰 체계이며 검증된 임상 척도가 아니다.

## 코드 수준의 검증

1. Pydantic이 필드·길이·허용 enum·추가 필드 금지를 검사한다.
2. `evidence_quote`는 실제로 전달한 전사 안에 존재하는 정확한 연속 부분 문자열이어야 한다.
3. `uncertain` 외의 유형에는 비어 있지 않은 근거 인용이 필요하다.
4. 종합 피드백의 `metric_id`는 실제 규칙 결과에 존재해야 한다.
5. 측정 상태가 `valid`가 아닌 지표를 장점 근거로 사용하면 결과를 거부한다.
6. 최종 JSON의 수치 점수는 로컬 `RuleBasedReport`에서 복사한다. LLM 출력에는 점수 필드가 없다.

스키마 검증은 문장의 사실성·교육적 타당성을 자동 보장하지 않는다. 특히 원문에 있는 인용이라고 해서 그 해석까지 정확하다는 뜻은 아니다. `services/ai/tests/test_realtime_language.py`는 입력·출력 계약을 검증하며, 실제 모델의 성능 평가는 별도 데이터셋이 필요하다.

## 실행 제어와 재현

`--llm`을 지정한 경우만 호출한다. `OPENAI_API_KEY`와 실제 사용 가능한 구조화 출력 모델 ID를 `OPENAI_MODEL` 또는 `--model`로 지정한다. 모델은 코드에 임의의 최신 이름으로 고정하지 않는다.

- 기본 요청 제한 시간 15초, SDK 자동 재시도 0회
- 최대 응답 토큰 2,000
- `store=False`
- 발화 전사 입력 최대 3,000자
- 최근 문맥 최대 4턴
- LLM 대기 큐 최대 4개; 초과한 새 작업은 `dropped`
- 프롬프트 4개 파일의 SHA-256과 선택 모델을 세션 JSON에 기록

학습자에게 제공되는 원시 측정과 점수는 API 실패와 관계없이 로컬 결과에 남는다. API 실패는 해당 발화·종합 피드백의 상태 필드에서 확인한다.

API 호출 없이 템플릿과 스키마를 검토하려면:

```bash
python services/ai/scripts/render_realtime_prompts.py
```

`outputs/prompt_preview/request_preview.json`에 렌더링된 예시 요청과 출력 JSON Schema가 생성된다. API 키나 실제 사용자 데이터는 필요하지 않다.

## 후속 실험 설계

새 프롬프트를 채택하기 전에는 사람이 라벨링한 발화 집합에서 다음을 비교한다.

| 평가 항목 | 방법 |
|---|---|
| 발화 유형 분류 | 유형별 precision·recall·macro F1, 혼동 행렬 |
| 근거 인용 | 원문 부분 문자열 일치율; 현재 코드는 불일치 결과를 거부 |
| 근거 없는 해석 | 사람이 검토한 비근거 진술 비율 |
| 판단 유보 | 애매한 문장에서 `uncertain`의 적절성 |
| 지연 | 발화 종료부터 STT·LLM 완료까지 p50·p95 |
| 견고성 | 명령문이 들어간 발화, 빈 발화, 전사 오류·긴 발화에 대한 계약 준수 |

현재 저장된 합성 통합 시험 결과를 위 모델 성능 지표로 보고하지 않는다.
