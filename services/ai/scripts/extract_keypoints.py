#!/usr/bin/env python3
"""Command-line entry point for local Pose and FaceMesh extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from care_hero.errors import CareHeroError  # noqa: E402
from care_hero.pipeline import extract_video_keypoints, save_keypoints_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose and FaceMesh landmarks from a local video."
    )
    parser.add_argument("--video", required=True, help="Input mp4, mov, or avi path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def show_progress(current: int, total: int) -> None:
        print(f"\rkeypoint extraction: {current}/{total}", end="", flush=True)

    try:
        result = extract_video_keypoints(args.video, progress_callback=show_progress)
        output_path = save_keypoints_json(result, args.output)
    except CareHeroError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print()
    print(f"analyzed frames: {result.analyzed_frame_count}")
    print(f"detected frames: {result.detected_frame_count}")
    print(f"saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
