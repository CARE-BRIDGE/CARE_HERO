# Reference code 안내

이 폴더는 전체 프로젝트 코드보다 짧은 최소 예제를 제공한다.

- `01_mediapipe_pose_face.py`: 한 이미지에서 Pose와 FaceMesh를 검출하는 최소 코드
- `02_pipeline_order.py`: 입력부터 보고서까지 데이터가 어떻게 이동하는지 보여 주는 의사 실행 코드
- `copy_mediapipe_weights.py`: 설치된 MediaPipe 패키지에서 사용하는 사전학습 weight를 이 폴더의 `weights/mediapipe/`로 복사

공식 참고 자료:

- MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Logistic regression 개념: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

MediaPipe weight는 Google이 학습한 사전학습 모델이고, `train.py`는 그 모델을 다시 학습하지 않는다. `train.py`가 학습하는 대상은 MediaPipe 결과에서 계산한 특징을 사용하는 작은 보조 분류 모델이다.

