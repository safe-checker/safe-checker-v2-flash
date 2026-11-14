#!/usr/bin/env python3
"""Build the sample set handed to the third-party test body.

The outline requires stratified sampling over 10 construction scenes, at
least 100 samples per scene and at least 1000 in total, with every sample
carrying an annotation. This script assembles the manifest from the
labelled image set plus video frames, and it reports exactly how many
samples each scene is short by.

The sampling rule is fixed before the run and printed with the result, so
the set can be audited. Nothing here picks "the images the system gets
right"; it fills scene quotas in a fixed order and reports the shortfall.

Usage:
    python acceptance_tests/build_sample_set.py \
        --image-set /path/to/施工安全图集_140张 \
        --frames-dir acceptance_tests/dataset/frames \
        --per-scene 100 --out acceptance_tests/dataset/sample_set
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import SCENES, scene_name, write_json, write_jsonl

SCAFFOLD_FAMILY = "SCAFFOLD_FAMILY"


def load_labels(image_set: Path) -> list[dict[str, Any]]:
    path = image_set / "labels.jsonl"
    if not path.exists():
        raise SystemExit(f"labels.jsonl not found under {image_set}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect(image_set: Path, frames_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}

    for row in load_labels(image_set):
        image = image_set / row["image_path"]
        if not image.exists():
            continue
        code = row.get("test_scene_code") or "UNCLASSIFIED"
        pools.setdefault(code, []).append({
            "sample_id": row["sample_id"],
            "scene_code": code,
            "scene": row.get("test_scene", ""),
            "image_path": str(image),
            "source": "image_set",
            "annotation": row.get("inspection_item", ""),
            "labels": row.get("labels") or [],
        })

    if frames_dir and frames_dir.exists():
        for scene_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
            for image in sorted(scene_dir.iterdir()):
                if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                pools.setdefault(scene_dir.name, []).append({
                    "sample_id": f"{scene_dir.name}-{image.stem}",
                    "scene_code": scene_dir.name,
                    "scene": scene_name(scene_dir.name),
                    "image_path": str(image),
                    "source": "video_frame",
                    "annotation": "",
                    "labels": [],
                })
    return pools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-set", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--per-scene", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/dataset/sample_set"))
    parser.add_argument("--scaffold-family-scene", default="SCAFFOLD_COUPLER_TYPE")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    pools = collect(args.image_set, args.frames_dir)

    picked: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for code in sorted(SCENES):
        pool = list(pools.get(code, []))
        # 家族标注的脚手架图只归入一个型式，不复制到四个型式里充数。
        if not pool and code == args.scaffold_family_scene:
            pool = list(pools.get(SCAFFOLD_FAMILY, []))
        rng.shuffle(pool)
        taken = pool[: args.per_scene]
        for sample in taken:
            sample = dict(sample)
            sample["scene_code"] = code
            sample["scene"] = scene_name(code)
            picked.append(sample)
        report.append({
            "scene_code": code, "scene": scene_name(code),
            "available": len(pool), "target": args.per_scene, "taken": len(taken),
            "shortfall": max(0, args.per_scene - len(taken)),
        })

    total_target = args.per_scene * len(SCENES)
    shortfall = total_target - len(picked)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "sample_set.jsonl", picked)
    write_json(args.out / "sample_set_report.json", {
        "sampling_rule": "按场景配额分层抽样；场景内用固定随机种子打乱后取前 N；"
                         "抽样规则在跑测前固定并随数据集一并归档。",
        "seed": args.seed,
        "per_scene_target": args.per_scene,
        "total_target": total_target,
        "total_taken": len(picked),
        "total_shortfall": shortfall,
        "meets_minimum": shortfall == 0,
        "scenes_covered": len({s["scene_code"] for s in picked}),
        "per_scene": report,
        "sources": {"image_set": str(args.image_set), "frames_dir": str(args.frames_dir)},
    })

    print(f"样本集：{args.out / 'sample_set.jsonl'}  共 {len(picked)} 条，目标 {total_target} 条，缺口 {shortfall}")
    for row in report:
        flag = "OK " if row["shortfall"] == 0 else "缺口"
        print(f"  [{flag}] {row['scene_code']:<26}{row['scene']:<12}{row['taken']}/{row['target']}")
    if shortfall:
        print("\n样本量未达标：请补充视频抽帧后重新生成，不要用重复样本补足。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
