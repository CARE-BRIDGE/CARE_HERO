"""설치된 MediaPipe 패키지에서 실제 사전학습 weight를 복사한다."""

from __future__ import annotations

import shutil
from pathlib import Path

import mediapipe


FILES = (
    "modules/pose_detection/pose_detection.tflite",
    "modules/pose_landmark/pose_landmark_full.tflite",
    "modules/face_detection/face_detection_short_range.tflite",
    "modules/face_landmark/face_landmark_with_attention.tflite",
)


def main() -> None:
    package_root = Path(mediapipe.__file__).resolve().parent
    output_root = Path(__file__).resolve().parents[1] / "weights" / "mediapipe"
    output_root.mkdir(parents=True, exist_ok=True)
    for relative_path in FILES:
        source = package_root / relative_path
        destination = output_root / source.name
        if not source.is_file():
            raise FileNotFoundError(f"MediaPipe weight를 찾지 못했습니다: {source}")
        shutil.copy2(source, destination)
        print(f"복사: {destination}")


if __name__ == "__main__":
    main()

