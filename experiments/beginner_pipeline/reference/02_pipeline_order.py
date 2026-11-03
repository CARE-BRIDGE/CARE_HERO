"""전체 파이프라인의 역할과 순서를 한 화면에서 이해하기 위한 참고 코드."""


def care_hero_pipeline(input_media):
    frames = load_and_sample_frames(input_media)       # 1. 입력 읽기
    pose = detect_pose(frames)                         # 2. 몸 관절 33개
    face = detect_face_mesh(frames)                    # 3. 얼굴 좌표 최대 478개
    features = calculate_explainable_features(pose)    # 4. 검출률·정면 비율·기울기
    prediction = behavior_model(features)              # 5. 예제 보조 모델
    speech_text = optional_stt(input_media)             # 6. 선택: 발화 내용 전사
    return make_report(features, prediction, speech_text)  # 7. JSON 결과


# 실제로 실행되는 구현은 상위 폴더의 pipeline.py와 run_pipeline.py에 있다.

