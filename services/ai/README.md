# CARE:HERO AI workspace

자세·얼굴 좌표 검출, STT, 규칙 평가, LLM 템플릿과 실시간 작업 흐름을 담당합니다. 기존 AI 코드를 이 폴더로 모았으며, 처리 알고리즘은 변경하지 않았습니다.

```text
ai/
├── pyproject.toml   # AI Python 패키지·선택 의존성
├── src/care_hero/   # 모델 어댑터·평가·실시간 처리
├── scripts/         # 실행·프롬프트 검토·구조도 생성
├── configs/         # 모델·큐·시나리오 설정
├── prompts/         # 버전 관리하는 LLM 템플릿
└── tests/           # AI 단위·통합 테스트
```

다음 명령은 **저장소 루트에서** 실행합니다.

```bash
source .venv/bin/activate
python -m pip install -e './services/ai[dev,realtime]'
python services/ai/scripts/run_realtime.py --demo --duration 4
python -m pytest
```

실제 장치 입력:

```bash
python services/ai/scripts/run_realtime.py --camera 0 --audio
```

설정·프롬프트는 이 AI workspace 안에서 찾으며, 기본 결과는 저장소 루트의 `outputs/`에 저장합니다. 공통 구조도는 루트 `docs/figures/`에 생성합니다. 루트 `pyproject.toml`은 테스트 진입점이고, 설치할 패키지 설정은 이 폴더의 `pyproject.toml`입니다.

[프로젝트 README](../../README.md) · [코드·출력 해설](../../docs/realtime_code_walkthrough.md) · [LLM engineering](../../docs/realtime_llm_engineering.md) · [협업 구조](../../docs/team_structure.md)
