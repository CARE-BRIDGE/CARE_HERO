# Shared integration contracts

프론트엔드·백엔드·AI가 함께 합의한 데이터 정의를 보관하는 작업 공간입니다. 현재는 폴더와 협업 규칙만 마련했으며, HTTP/WebSocket API와 공통 스키마는 아직 확정하지 않았습니다.

```text
contracts/
├── schemas/    # 합의된 요청·응답·이벤트 JSON Schema
└── examples/   # 해당 스키마를 만족하는 비식별 예시
```

합의할 때에는 세션 ID, 이벤트 순서, 미디어 시간과 생성 시간, 상태·오류 값, `null`의 의미, 점수·단위, 스키마 버전을 함께 정합니다. 전송 방식이 정해지면 OpenAPI 또는 실시간 이벤트 정의도 이 위치에 추가할 수 있습니다.

현재 AI의 내부 구조는 [`services/ai/src/care_hero/schemas.py`](../../services/ai/src/care_hero/schemas.py)와 [실시간 출력 해설](../../docs/realtime_code_walkthrough.md)에 있습니다. 이 파일들은 협의의 참고 자료이며 공통 API 계약과 구분합니다.

공통 필드 변경은 세 담당자가 함께 검토하고, `schemas/`와 `examples/`를 같은 변경에서 갱신합니다. 서비스 내부용 Python 타입은 AI 폴더에, 프론트 화면 전용 타입은 프론트 폴더에 둡니다.
