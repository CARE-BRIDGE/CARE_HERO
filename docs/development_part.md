# CARE:HERO 개발 파트

작성일: 2026-08-05

## 개발 목표

CARE:HERO의 개발 목표는 사용자의 연습 영상을 로컬 환경에서 분석하여 자세
landmark, 규칙 기반 측정값, 음성 전사 결과를 구조화된 리포트로 제공하는 것임.
측정 결과는 Python 규칙 코드가 결정하며 LLM은 선택적으로 설명 문장만 생성하는
구조임.

## 전체 파이프라인

```text
영상 입력
  → OpenCV 프레임 추출
  → MediaPipe Pose keypoint 추출
  → 신체 노출 범위 분류
  → Rule-based measurement
  → 선택적 로컬 STT
  → JSON·Markdown 리포트
  → 선택적 LLM 피드백
```

영상, Pose, scoring, STT, report 모듈을 분리하여 한 기능의 실패가 다른 분석 결과를
폐기하지 않도록 구성하였음.

## 영상 및 Keypoint 분석

`loaders.py`에서 영상의 확장자, FPS, 프레임 수, 해상도를 검증하고 프레임을 순차적으로
디코딩함. `pose.py`에서는 MediaPipe Pose를 영상당 한 번 초기화하고 각 프레임에서
33개 landmark의 normalized x·y 좌표와 visibility를 추출함.

좌표값은 이미지 내부의 상대 위치이며 실제 거리나 물리적 속도를 의미하지 않음.
검출 실패 시 빈 좌표를 정상값으로 대체하지 않고 `detected=false`로 기록함.

## 상반신 영상 대응

카메라에 전신이 포함되지 않는 경우를 처리하기 위해 landmark visibility 기반의 신체
노출 범위 분류를 추가하였음.

- `full_body`: 코·어깨와 다리 landmark가 지속적으로 확인되는 영상임.
- `upper_body`: 코·어깨는 확인되지만 무릎·발목이 대부분 보이지 않는 영상임.
- `mixed`: 프레임별 노출 범위가 일정하지 않거나 핵심 landmark가 부분적으로 누락된 영상임.
- `detection_failed`: 분석 가능한 Pose가 검출되지 않은 영상임.

상반신 영상에서도 코·어깨 기반 정면 배치 지표는 계산함. 골반 landmark가 없는 경우
몸통 정렬과 안정성은 `not_applicable`로 반환하며 전체 총점은 생성하지 않음. 이를 통해
누락된 신체 정보를 0점이나 만점으로 잘못 해석하지 않도록 하였음.

## 추가 모달리티: 로컬 STT

영상 속 발화를 텍스트로 변환하기 위한 선택적 로컬 STT 모듈을 추가하였음.
`faster-whisper`를 optional dependency로 사용하며 `--stt` 옵션이 있을 때만 모델을
초기화함.

STT 결과에는 언어, 모델 정보, 전체 텍스트와 구간별 시작·종료 시각이 포함됨. STT
초기화나 전사에 실패해도 Pose 분석과 규칙 리포트는 유지하며, 오류 상태만 별도로
기록하도록 구성하였음.

향후 음성 지표를 추가할 경우 다음 항목을 별도 Metric으로 확장할 수 있음.

- 발화 시작까지 걸린 시간
- 발화 구간과 침묵 구간의 비율
- 필수 문장 또는 핵심 표현의 포함 여부
- 행동 구간과 발화 구간의 시간적 순서

현재 단계에서는 음성 내용을 의료 판단, 감정 인식 또는 사용자 적합성 판단에 사용하지
않음.

## Rule-based Scoring

현재 규칙 엔진은 Pose 검출 품질, 2D 정면 자세 추정 비율, 몸통 수직 정렬, 몸통 기울기
변동을 provisional 지표로 계산함. 모든 threshold와 weight는 시나리오 YAML에서
관리함.

필수 데이터가 부족하면 항목 상태를 `detection_failed`, `insufficient_data` 또는
`not_applicable`로 반환함. 모든 활성 지표가 유효한 경우에만 가중 총점을 생성함.

## 리포트 및 LLM 경계

최종 리포트는 JSON과 Markdown으로 생성됨. JSON은 시스템 연계용 구조 데이터이며
Markdown은 개발자와 기획자가 결과를 직접 확인하기 위한 문서임.

LLM을 사용할 경우 frame 이미지나 전체 keypoint sequence를 전달하지 않고 규칙 엔진이
확정한 compact report만 전달함. LLM 출력은 Pydantic 스키마로 검증하며 원시 측정값이나
규칙 점수를 변경할 수 없음.

## 검증 전략

- 실제 사용자 영상 대신 synthetic landmark 중심의 단위 테스트를 우선함.
- 전신·상반신·검출 실패·낮은 visibility 조건을 각각 테스트함.
- FPS와 해상도가 달라도 normalized 좌표 기반 결과가 유지되는지 확인함.
- STT 테스트에서는 실제 모델 대신 fake backend를 주입하여 네트워크와 모델 다운로드 없이 검증함.
- 한 Metric 또는 STT가 실패해도 다른 분석 결과가 유지되는지 확인함.

## 현재 구현 상태

| 개발 항목 | 상태 | 관련 코드 |
|---|---|---|
| 영상 로드 및 프레임 추출 | 구현 완료 | `loaders.py` |
| MediaPipe Pose keypoint | 구현 완료 | `pose.py` |
| 전신·상반신 노출 범위 분류 | 구현 완료 | `coverage.py` |
| Rule-based scoring | provisional 구현 | `scoring.py` |
| 로컬 STT 인터페이스 | 구현 완료 | `stt.py` |
| JSON·Markdown 리포트 | 구현 완료 | `reporting.py` |
| 실제 STT 모델 실행 검증 | 환경 설치 후 검증 필요 | `faster-whisper` |
| 6대 행동 지표 | 정의 및 전문가 기준 필요 | 시나리오 YAML 확장 예정 |

## 다음 개발 범위

1. 실제 상반신·전신 영상을 이용한 coverage 판정 검증이 필요함.
2. 로컬 Whisper 모델별 처리 속도와 한국어 전사 품질 비교가 필요함.
3. 행동 지표별 측정식, 임계값, 가중치의 전문가 검토가 필요함.
4. Pose와 발화 구간을 결합하는 multimodal timestamp 정렬이 필요함.
5. 분석 백엔드 검증 이후 사용자 입력 화면과 결과 시각화 연결이 필요함.
