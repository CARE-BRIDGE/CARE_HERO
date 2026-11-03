"""CARE:HERO의 입력부터 결과 저장까지 연결한 초보자용 전체 파이프라인."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import mediapipe as mp

from features import (
    FrameObservation,
    Landmark,
    calculate_front_facing,
    calculate_torso_tilt,
    classify_coverage,
    summarize_observations,
)
from feedback import build_feedback
from model import BeginnerBehaviorModel
from stt import transcribe_video


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else Path(__file__).with_name("config.json")
    return json.loads(config_path.read_text(encoding="utf-8"))


def run_analysis(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    model_path: str | Path | None = None,
    sample_every: int = 5,
    use_stt: bool = False,
    whisper_model_size: str = "small",
) -> dict[str, Any]:
    """미디어를 분석하고 JSON과 첫 프레임 시각화 이미지를 저장한다."""

    media_path = Path(input_path)
    if not media_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {media_path}")
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()

    observations: list[FrameObservation] = []
    frame_details: list[dict[str, Any]] = []
    input_metadata: dict[str, Any] | None = None
    preview_saved = False

    with _load_detectors(config) as (pose_model, face_model):
        for frame_index, timestamp, bgr_frame, metadata in _iter_media_frames(
            media_path, config, sample_every
        ):
            if input_metadata is None:
                input_metadata = metadata
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            pose_result = pose_model.process(rgb_frame)
            face_result = face_model.process(rgb_frame)

            landmarks = _pose_landmark_mapping(pose_result)
            pose_detected = bool(landmarks)
            face_landmark_count = _face_landmark_count(face_result)
            coverage = (
                classify_coverage(
                    landmarks,
                    minimum_visibility=config["minimum_landmark_visibility"],
                )
                if pose_detected
                else "detection_failed"
            )
            front_facing = calculate_front_facing(
                landmarks,
                minimum_visibility=config["minimum_landmark_visibility"],
                maximum_offset=config["front_facing_maximum_offset"],
            )
            torso_tilt = calculate_torso_tilt(
                landmarks,
                minimum_visibility=config["minimum_landmark_visibility"],
            )
            observations.append(
                FrameObservation(
                    pose_detected=pose_detected,
                    face_landmark_count=face_landmark_count,
                    landmarks=landmarks,
                    coverage=coverage,
                    front_facing=front_facing,
                    torso_tilt_degrees=torso_tilt,
                )
            )
            frame_details.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp, 3),
                    "pose_detected": pose_detected,
                    "valid_pose_coordinate_count": sum(
                        point.visibility >= config["minimum_landmark_visibility"]
                        for point in landmarks.values()
                    ),
                    "total_pose_coordinate_count": 33 if pose_detected else 0,
                    "face_landmark_count": face_landmark_count,
                    "coverage": coverage,
                    "front_facing": front_facing,
                    "torso_tilt_degrees": (
                        round(torso_tilt, 3) if torso_tilt is not None else None
                    ),
                }
            )

            if not preview_saved:
                annotated = _draw_preview(bgr_frame, pose_result, face_result)
                cv2.imwrite(str(output_dir / "annotated_preview.jpg"), annotated)
                preview_saved = True

    summary = summarize_observations(observations)
    feature_values = summary["features"]
    if not isinstance(feature_values, dict):
        raise RuntimeError("특징 요약 형식이 올바르지 않습니다.")

    selected_model_path = (
        Path(model_path)
        if model_path
        else Path(__file__).parent / "weights" / "behavior_model.json"
    )
    improvement_probability: float | None = None
    if selected_model_path.is_file():
        behavior_model = BeginnerBehaviorModel.load(selected_model_path)
        improvement_probability = behavior_model.predict_probability(feature_values)

    result: dict[str, Any] = {
        "input": {
            "path": str(media_path.resolve()),
            **(input_metadata or {}),
        },
        "pipeline": {
            "pose": "MediaPipe Pose, 33 landmarks",
            "face": "MediaPipe FaceMesh, refine_landmarks=True, up to 478 landmarks",
            "behavior_model": "NumPy logistic regression" if improvement_probability is not None else "not_run",
            "stt": "faster-whisper" if use_stt else "not_run",
        },
        "summary": summary,
        "behavior_model": {
            "status": "example_only" if improvement_probability is not None else "not_run",
            "needs_improvement_probability": (
                round(improvement_probability, 4)
                if improvement_probability is not None
                else None
            ),
            "weight_path": str(selected_model_path),
            "warning": "합성 예제 데이터로 학습한 시연값이며 공식 교육 점수나 의료 진단이 아님",
        },
        "feedback": build_feedback(
            feature_values,
            str(summary["analysis_scope"]),
            improvement_probability,
        ),
        "frames": frame_details,
    }

    if use_stt:
        if media_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("STT는 음성이 포함된 영상 파일에만 사용할 수 있습니다.")
        result["stt"] = transcribe_video(media_path, whisper_model_size)

    output_json = output_dir / "analysis.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


@contextmanager
def _load_detectors(config: dict[str, Any]) -> Iterator[tuple[Any, Any]]:
    """모델을 프레임마다 다시 만들지 않고 분석 세션 동안 한 번만 연다."""

    pose_model = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=int(config["pose_model_complexity"]),
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=float(config["minimum_detection_confidence"]),
        min_tracking_confidence=float(config["minimum_tracking_confidence"]),
    )
    face_model = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=bool(config["refine_face_landmarks"]),
        min_detection_confidence=float(config["minimum_detection_confidence"]),
        min_tracking_confidence=float(config["minimum_tracking_confidence"]),
    )
    try:
        yield pose_model, face_model
    finally:
        pose_model.close()
        face_model.close()


def _iter_media_frames(
    path: Path, config: dict[str, Any], sample_every: int
) -> Iterator[tuple[int, float, Any, dict[str, Any]]]:
    if sample_every < 1:
        raise ValueError("sample_every는 1 이상이어야 합니다.")
    extension = path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("이미지를 읽지 못했습니다.")
        original_height, original_width = frame.shape[:2]
        resized = _resize_frame(frame, config)
        height, width = resized.shape[:2]
        yield 0, 0.0, resized, {
            "media_type": "image",
            "original_resolution": f"{original_width}x{original_height}",
            "analysis_resolution": f"{width}x{height}",
        }
        return
    if extension not in VIDEO_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("영상을 열지 못했습니다.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                resized = _resize_frame(frame, config)
                height, width = resized.shape[:2]
                yield frame_index, frame_index / fps, resized, {
                    "media_type": "video",
                    "original_resolution": f"{original_width}x{original_height}",
                    "analysis_resolution": f"{width}x{height}",
                    "fps": fps,
                    "total_frame_count": total_frames,
                    "sample_every": sample_every,
                }
            frame_index += 1
    finally:
        capture.release()


def _resize_frame(frame: Any, config: dict[str, Any]) -> Any:
    height, width = frame.shape[:2]
    scale = min(
        1.0,
        float(config["maximum_analysis_width"]) / width,
        float(config["maximum_analysis_height"]) / height,
    )
    if scale >= 1.0:
        return frame
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def _pose_landmark_mapping(result: Any) -> dict[str, Landmark]:
    if result.pose_landmarks is None:
        return {}
    output: dict[str, Landmark] = {}
    for landmark_type in mp.solutions.pose.PoseLandmark:
        point = result.pose_landmarks.landmark[landmark_type.value]
        output[landmark_type.name.lower()] = Landmark(
            x=float(point.x),
            y=float(point.y),
            visibility=float(point.visibility),
        )
    return output


def _face_landmark_count(result: Any) -> int:
    if not result.multi_face_landmarks:
        return 0
    return len(result.multi_face_landmarks[0].landmark)


def _draw_preview(frame: Any, pose_result: Any, face_result: Any) -> Any:
    preview = frame.copy()
    drawing = mp.solutions.drawing_utils
    if pose_result.pose_landmarks is not None:
        drawing.draw_landmarks(
            preview,
            pose_result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 190, 210), thickness=2, circle_radius=2),
            connection_drawing_spec=drawing.DrawingSpec(color=(0, 150, 180), thickness=2),
        )
    if face_result.multi_face_landmarks:
        drawing.draw_landmarks(
            preview,
            face_result.multi_face_landmarks[0],
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing.DrawingSpec(color=(20, 190, 150), thickness=1),
        )
    return preview

