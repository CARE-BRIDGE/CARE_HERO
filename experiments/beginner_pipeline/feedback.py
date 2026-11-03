"""숫자 결과를 초보자도 이해하기 쉬운 문장으로 바꾼다."""

from __future__ import annotations

from typing import Mapping


def build_feedback(
    features: Mapping[str, float | None],
    analysis_scope: str,
    improvement_probability: float | None,
) -> list[str]:
    messages: list[str] = []

    pose_rate = features.get("pose_detection_rate")
    if pose_rate is not None and pose_rate < 0.7:
        messages.append("몸 관절 검출률이 낮습니다. 전신이 화면 안에 보이도록 조명과 카메라 위치를 확인해 주세요.")
    else:
        messages.append("분석에 필요한 몸 관절 좌표가 비교적 안정적으로 검출되었습니다.")

    front_ratio = features.get("front_facing_ratio")
    if front_ratio is None:
        messages.append("얼굴과 양쪽 어깨 좌표가 부족해 정면 여부를 판단하지 않았습니다.")
    elif front_ratio < 0.7:
        messages.append("카메라 정면을 유지한 프레임이 적었습니다. 상대를 바라보듯 얼굴과 어깨 방향을 맞춰 보세요.")
    else:
        messages.append("대부분의 프레임에서 얼굴과 어깨가 카메라 정면을 향했습니다.")

    if analysis_scope == "upper_body":
        messages.append("상반신 촬영으로 확인 가능한 항목만 사용했고 하반신 항목은 점수에 넣지 않았습니다.")

    if improvement_probability is not None:
        percent = round(improvement_probability * 100)
        messages.append(
            f"예제 학습 모델의 개선 필요 가능성은 {percent}%입니다. 이 값은 합성 데이터로 만든 시연값이며 공식 점수가 아닙니다."
        )
    return messages

