#!/usr/bin/env python3
"""Extract sample frames from site video material.

The P7 latency test needs 1000 samples over 10 scenes. The labelled image
set only holds 140 pictures, so the rest comes from the site video the
integration team already collects. Frames are written into one directory
per scene code, which is what build_dataset.py expects.

Video decoding needs ffmpeg on PATH, or opencv-python if it is installed.
Neither is a runtime dependency of the detector, so this script keeps the
dependency optional and fails with a clear message instead of a traceback.

Usage:
    python acceptance_tests/extract_frames.py \
        --video /path/to/site.mp4 --scene WORK_AT_HEIGHT \
        --out acceptance_tests/dataset/frames --every 5 --max 120
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import SCENES, scene_name


def decode_with_ffmpeg(video: Path, out_dir: Path, every: int, max_frames: int,
                       width: int) -> list[Path]:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise SystemExit(
            "ffmpeg 未找到。抽帧需要系统安装 ffmpeg（macOS: brew install ffmpeg），"
            "或在虚拟环境中安装 opencv-python 后重试。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.jpg"
    command = [
        binary, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps=1/{every},scale={width}:-2",
        "-frames:v", str(max_frames),
        "-q:v", "3",
        str(pattern),
    ]
    subprocess.run(command, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def decode_with_opencv(video: Path, out_dir: Path, every: int, max_frames: int,
                       width: int) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ffmpeg 未找到且 opencv-python 未安装，二者任选其一即可抽帧。"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频：{video}")
    written: list[Path] = []
    index = 0
    while len(written) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            height = int(frame.shape[0] * width / frame.shape[1])
            resized = cv2.resize(frame, (width, height))
            target = out_dir / f"frame_{len(written):06d}.jpg"
            cv2.imwrite(str(target), resized)
            written.append(target)
        index += 1
    capture.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input video file.")
    parser.add_argument("--scene", required=True, choices=sorted(SCENES),
                        help="Scene code the frames belong to.")
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/dataset/frames"))
    parser.add_argument("--every", type=int, default=5, help="Take one frame every N seconds.")
    parser.add_argument("--max", dest="max_frames", type=int, default=100)
    parser.add_argument("--width", type=int, default=1280, help="Frame width in pixels.")
    args = parser.parse_args()

    out_dir = args.out / args.scene
    if shutil.which("ffmpeg"):
        frames = decode_with_ffmpeg(args.video, out_dir, args.every, args.max_frames, args.width)
    else:
        frames = decode_with_opencv(args.video, out_dir, args.every, args.max_frames, args.width)

    print(f"场景 {args.scene}（{scene_name(args.scene)}）抽出 {len(frames)} 帧 -> {out_dir}")
    if len(frames) < args.max_frames:
        print("提示：帧数未达上限，视频可能较短或 --every 间隔过大。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
