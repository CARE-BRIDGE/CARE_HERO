"""명령 한 줄로 전체 분석 파이프라인을 실행한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="이미지/영상에서 Pose·FaceMesh 특징을 계산하고 결과를 저장합니다."
    )
    parser.add_argument("input", help="분석할 이미지 또는 영상 경로")
    parser.add_argument(
        "--output",
        help="결과 폴더. 생략하면 outputs/<입력파일명>/ 사용",
    )
    parser.add_argument(
        "--weights",
        help="behavior_model.json 경로. 생략하면 weights/behavior_model.json 사용",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="영상에서 몇 프레임마다 1장을 분석할지 지정 (기본 5)",
    )
    parser.add_argument("--stt", action="store_true", help="영상 발화를 faster-whisper로 전사")
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper 모델 크기 (기본 small)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    default_output = Path(__file__).parent / "outputs" / input_path.stem
    result = run_analysis(
        input_path,
        args.output or default_output,
        model_path=args.weights,
        sample_every=args.sample_every,
        use_stt=args.stt,
        whisper_model_size=args.whisper_model,
    )
    output_dir = Path(args.output or default_output).resolve()
    print("분석을 완료했습니다.")
    print(f"- 분석 범위: {result['summary']['analysis_scope']}")
    print(f"- JSON 결과: {output_dir / 'analysis.json'}")
    print(f"- 시각화 결과: {output_dir / 'annotated_preview.jpg'}")


if __name__ == "__main__":
    main()

