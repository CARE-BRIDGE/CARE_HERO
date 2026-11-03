"""공식 MediaPipe Solutions API의 핵심 사용법만 남긴 최소 참고 코드."""

from __future__ import annotations

import sys

import cv2
import mediapipe as mp


def main(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 읽지 못했습니다.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(static_image_mode=True) as pose, \
         mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face:
        pose_result = pose.process(rgb)
        face_result = face.process(rgb)

    pose_count = len(pose_result.pose_landmarks.landmark) if pose_result.pose_landmarks else 0
    face_count = len(face_result.multi_face_landmarks[0].landmark) if face_result.multi_face_landmarks else 0
    print(f"Pose 좌표 수: {pose_count}")
    print(f"FaceMesh 좌표 수: {face_count}")


if __name__ == "__main__":
    main(sys.argv[1])

