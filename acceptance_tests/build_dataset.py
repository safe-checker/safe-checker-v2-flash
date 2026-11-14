#!/usr/bin/env python3
"""Assemble the acceptance test dataset.

The P7 latency test needs at least 1000 samples spread over 10 construction
scenes. Two sources feed the manifest:

  --image-set   a labelled image set with the labels.jsonl layout
  --frames-dir  frames extracted from the site video material, one
                sub-directory per scene code

The script never invents samples. If a scene cannot reach the target it is
reported as a shortfall in the dataset report instead of being padded.

Usage:
    python acceptance_tests/build_dataset.py \
        --image-set /path/to/施工安全图集_140张 \
        --frames-dir /path/to/frames \
        --per-scene 100 \
        --out acceptance_tests/dataset
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import SCENES, scene_name, write_json, write_jsonl

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 服务端的上传上限，与 src/api_server.py 的 API_MAX_UPLOAD_BYTES 一致（12 MB）。
UPLOAD_LIMIT_BYTES = int(os.getenv("API_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))


def load_labels(image_set: Path) -> list[dict[str, Any]]:
    labels_path = image_set / "labels.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"labels.jsonl not found under {image_set}")
    rows: list[dict[str, Any]] = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def from_image_set(image_set: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in load_labels(image_set):
        image = image_set / row["image_path"]
        if not image.exists():
            continue
        samples.append({
            "sample_id": row["sample_id"],
            "scene_code": row.get("test_scene_code", ""),
            "scene": row.get("test_scene", ""),
            "image_path": str(image),
            "source": "image_set",
            "inspection_item": row.get("inspection_item", ""),
        })
    return samples


def from_frames(frames_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not frames_dir.exists():
        return samples
    for scene_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        code = scene_dir.name
        for image in sorted(scene_dir.iterdir()):
            if image.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            samples.append({
                "sample_id": f"{code}-{image.stem}",
                "scene_code": code,
                "scene": scene_name(code),
                "image_path": str(image),
                "source": "video_frame",
                "inspection_item": "",
            })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-set", type=Path, help="Directory holding labels.jsonl and images/.")
    parser.add_argument("--frames-dir", type=Path, help="Video frames, one sub-directory per scene code.")
    parser.add_argument("--per-scene", type=int, default=100, help="Target samples per scene.")
    parser.add_argument("--scaffold-family-scene", default="SCAFFOLD_COUPLER_TYPE",
                        help="脚手架图库只标到家族级，默认归入扣件式；其余三个型式需靠抽帧补足，"
                             "不复制同一批图片充数。")
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/dataset"))
    args = parser.parse_args()

    pool: list[dict[str, Any]] = []
    if args.image_set:
        pool.extend(from_image_set(args.image_set))
    if args.frames_dir:
        pool.extend(from_frames(args.frames_dir))
    if not pool:
        raise SystemExit("No samples found. Pass --image-set and/or --frames-dir.")

    # 超过服务端上传上限的图片无法通过接口送检，直接排除并记录，不留到执行阶段才发现。
    oversized: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []
    for sample in pool:
        size = os.path.getsize(sample["image_path"])
        sample["bytes"] = size
        if size > UPLOAD_LIMIT_BYTES:
            oversized.append({"sample_id": sample["sample_id"], "bytes": size,
                               "image_path": sample["image_path"]})
            continue
        usable.append(sample)
    pool = usable

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for sample in pool:
        code = sample["scene_code"] or "UNCLASSIFIED"
        by_scene.setdefault(code, []).append(sample)

    picked: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for code in sorted(SCENES):
        available = by_scene.get(code, [])
        # 脚手架图库只标了家族级场景。同一批图不能同时充当四个型式的样本，
        # 因此家族图只归入指定的一个型式，其余型式如实报缺口，等抽帧补足。
        if not available and code.startswith("SCAFFOLD_"):
            available = by_scene.get("SCAFFOLD_FAMILY", []) if code == args.scaffold_family_scene else []
        taken = available[: args.per_scene]
        for sample in taken:
            sample = dict(sample)
            sample["scene_code"] = code
            sample["scene"] = scene_name(code)
            picked.append(sample)
        report.append({
            "scene_code": code,
            "scene": scene_name(code),
            "available": len(available),
            "unique_available": len({s["image_path"] for s in available}),
            "target": args.per_scene,
            "taken": len(taken),
            "shortfall": max(0, args.per_scene - len(taken)),
        })

    for code, items in sorted(by_scene.items()):
        if code not in SCENES and code != "SCAFFOLD_FAMILY":
            report.append({
                "scene_code": code, "scene": code, "available": len(items),
                "target": 0, "taken": 0, "shortfall": 0, "note": "场景键不在约定 10 类内，已排除",
            })

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    write_jsonl(manifest_path, picked)
    total_shortfall = sum(r["shortfall"] for r in report)
    write_json(args.out / "dataset_report.json", {
        "target_per_scene": args.per_scene,
        "total_target": args.per_scene * len(SCENES),
        "total_taken": len(picked),
        "total_shortfall": total_shortfall,
        "meets_minimum": total_shortfall == 0,
        "per_scene": report,
        "sources": {
            "image_set": str(args.image_set) if args.image_set else None,
            "frames_dir": str(args.frames_dir) if args.frames_dir else None,
        },
        "upload_limit_bytes": UPLOAD_LIMIT_BYTES,
        "excluded_oversized": oversized,
    })
    print(f"manifest: {manifest_path}  samples={len(picked)}  shortfall={total_shortfall}")
    if oversized:
        print(f"超出上传上限（{UPLOAD_LIMIT_BYTES / 1024 / 1024:.0f} MB）已排除 {len(oversized)} 张：")
        for item in oversized:
            print(f"  {item['sample_id']}  {item['bytes'] / 1024 / 1024:.1f} MB")
    for row in report:
        flag = "OK " if row["shortfall"] == 0 else "缺口"
        print(f"  [{flag}] {row['scene_code']:<26} {row['scene']:<12} "
              f"{row['taken']}/{row['target']}")
    if total_shortfall:
        print("\n样本量未达标：按要求不得用重复样本或非本场景样本补足，"
              "请补充视频抽帧后重新生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
