# 데이터 형식

`demo_train.csv`와 `demo_eval.csv`는 코드가 바로 동작하는지 보여 주기 위해 사람이 만든 **합성 예제**입니다.

- `pose_detection_rate`: 전체 분석 프레임 중 Pose 검출 비율
- `face_detection_rate`: 전체 분석 프레임 중 FaceMesh 검출 비율
- `front_facing_ratio`: 코가 양 어깨 중앙 범위 안에 있던 프레임 비율
- `mean_abs_torso_tilt_deg`: 몸통 기울기 절댓값 평균(도)
- `torso_tilt_std_deg`: 몸통 기울기의 흔들림 정도(표준편차)
- `upper_body_ratio`: 상반신 촬영으로 판정된 프레임 비율
- `needs_improvement`: 전문가 라벨을 가정한 목표값. 0은 보완 필요 낮음, 1은 보완 필요 높음

빈 값은 해당 항목을 계산할 수 없다는 뜻입니다. 특히 상반신 촬영에서는 골반 좌표가 부족해 기울기 값이 비어 있을 수 있습니다. 실제 개발에서는 촬영 동의를 받은 영상에서 특징을 추출하고, 교육 전문가의 기준으로 라벨을 부여한 뒤 학습/평가 대상자를 분리해야 합니다.

