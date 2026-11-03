# Frontend workspace

교육생·강사 화면을 담당하는 작업 공간입니다. 프레임워크와 의존성은 프론트엔드 담당자가 정한 뒤 이 폴더에 추가합니다.

```text
frontend/
├── src/       # 화면, UI 구성 요소, 상태 관리, 백엔드 클라이언트
└── public/    # 공개 정적 자산
```

연동 데이터 정의는 [`packages/contracts`](../../packages/contracts/README.md)에 모읍니다. 화면이 AI 내부 Python 구조나 로컬 결과 파일 경로에 직접 의존하지 않도록, 백엔드가 제공하는 인터페이스를 사용합니다. 현재 백엔드 API와 브라우저 실시간 전송은 구현 전입니다.

담당 경계와 협업 절차: [폴더 구조 안내](../../docs/team_structure.md).
