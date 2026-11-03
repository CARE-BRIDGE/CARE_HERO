# Weight 파일 구분

## `behavior_model.json`

`train.py`가 `data/demo_train.csv`를 사용해 생성하는 CARE:HERO 예제 보조 모델입니다. 사람이 읽을 수 있도록 JSON으로 저장합니다. 현재 데이터가 합성이므로 공식 점수나 실제 성능 근거로 사용할 수 없습니다.

## `mediapipe/*.tflite`

설치된 `mediapipe==0.10.21` 패키지에 포함된 Google 사전학습 모델을 복사한 것입니다.

- `pose_detection.tflite`: 화면에서 사람 영역 탐색
- `pose_landmark_full.tflite`: 사람 영역에서 몸 관절 좌표 예측
- `face_detection_short_range.tflite`: 가까운 얼굴 영역 탐색
- `face_landmark_with_attention.tflite`: 얼굴 및 홍채를 포함한 정밀 좌표 예측

`pipeline.py`의 MediaPipe Solutions API는 설치된 패키지 내부의 동일 weight를 자동으로 사용합니다. 로컬 복사본은 모델 구성과 출처를 제출 폴더 안에서 확인할 수 있도록 보관합니다.

