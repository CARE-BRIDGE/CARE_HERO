# Backend workspace

사용자·훈련 세션·결과 저장과 프론트엔드/AI 연결을 담당하는 작업 공간입니다. 언어·프레임워크·데이터베이스 선택은 백엔드 담당자가 정한 뒤 이 폴더에 추가합니다.

```text
backend/
├── src/       # API, 세션 관리, 저장소, AI 호출 어댑터
└── tests/     # API·연동 테스트
```

현재 AI 실행 진입점은 [`services/ai`](../../services/ai/README.md)에 있습니다. AI 모델·평가 로직은 해당 폴더가 담당합니다. Python 함수를 직접 호출할지, 별도 AI worker/service와 통신할지는 배포 구조를 정하며 합의합니다.

요청·응답·이벤트 계약은 [`packages/contracts`](../../packages/contracts/README.md)에 모읍니다. 기존 AI 이벤트·세션 JSON은 내부 출력이므로 그대로 공개 API라고 가정하지 않습니다. 현재 API 서버는 구현 전입니다.

담당 경계와 협업 절차: [폴더 구조 안내](../../docs/team_structure.md).
