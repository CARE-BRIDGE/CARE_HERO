# 협업용 폴더 구조

```text
CARE_HERO/
├── apps/
│   ├── frontend/           # 프론트 담당: 화면·UI·백엔드 클라이언트
│   └── backend/            # 백엔드 담당: API·세션·저장소·AI 연결
├── services/
│   └── ai/                 # AI 담당: 모델·처리·평가·프롬프트·테스트
│       ├── src/care_hero/
│       ├── scripts/
│       ├── configs/
│       ├── prompts/
│       ├── tests/
│       └── pyproject.toml
├── packages/
│   └── contracts/          # 공동 검토: 요청·응답·이벤트의 데이터 계약
│       ├── schemas/
│       └── examples/
├── docs/                   # 공통 설계·코드 해설·구조도
├── experiments/            # 독립적인 연구·학습 예제
├── outputs/                # 로컬 실행 결과; 버전 관리 제외
├── .venv/                  # 기존 로컬 Python 환경; 버전 관리 제외
├── pyproject.toml          # 저장소 루트의 테스트 설정
└── README.md               # 프로젝트 진입 문서
```

## 담당 경계

| 담당 | 주 작업 위치 | 책임 |
|---|---|---|
| 프론트엔드 | `apps/frontend/` | 입력 화면, 진행 상태 표시, 리포트 화면, 백엔드 연동 |
| 백엔드 | `apps/backend/` | 사용자·훈련 세션, API, 저장, AI 작업 요청과 결과 전달 |
| AI | `services/ai/` | 검출·STT·평가·LLM 처리, 입력/출력의 의미와 신뢰도 |
| 공동 | `packages/contracts/` | 요청·응답·실시간 이벤트 계약과 호환성 |
| 공동 | `docs/` | 서비스 구조, 연동 방식, 운영·테스트 절차 |

서비스 연동의 목표 흐름은 `Frontend → Backend → AI → Backend → Frontend`입니다. 현재 구현된 실행 경로는 AI의 로컬 카메라·마이크 CLI입니다. 브라우저 미디어 전송, 백엔드 API, AI worker 통신은 이후 함께 결정할 범위입니다.

## 협업할 때 먼저 맞출 항목

새 연동을 시작할 때 `packages/contracts/`에서 입력 단위, 세션 식별자, 상태·오류, 시간 기준, 지표 단위와 결측값을 합의합니다. 프론트는 그 계약에 맞춰 표시하고, 백엔드는 세션·전송을 연결하며, AI는 약속한 의미의 결과를 제공합니다. 각 파트의 내부 구현은 담당 폴더에서 독립적으로 관리합니다.

현재 AI의 `events.jsonl`, `windows.jsonl`, `session.json`은 참고할 내부 산출물입니다. 새 HTTP/WebSocket 계약으로 확정하기 전까지 프론트 화면을 해당 파일 경로에 직접 연결하지 않습니다.

## 이동 내역

| 이전 | 현재 |
|---|---|
| `src/` | `services/ai/src/` |
| `scripts/` | `services/ai/scripts/` |
| `configs/` | `services/ai/configs/` |
| `prompts/` | `services/ai/prompts/` |
| `tests/` | `services/ai/tests/` |
| AI 패키지 `pyproject.toml` | `services/ai/pyproject.toml` |
| `beginner_pipeline/` | `experiments/beginner_pipeline/` |

AI 패키지명 `care_hero`와 모듈 내부 import는 동일합니다. 폴더 이동 이후에는 루트 가상환경에서 `python -m pip install -e './services/ai[dev,realtime]'`로 editable 설치 경로를 갱신합니다. 실행 파일 경로와 문서 링크는 새 구조에 맞춰 갱신했습니다.
