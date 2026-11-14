#!/usr/bin/env python3
"""A/B arms for accuracy: merge rules and per-scene image resolution.

The arms are run interleaved on the same images inside one time window,
because upstream latency and the model itself drift over hours, so a
cross-window comparison is not a comparison.

Usage:
    .venv/bin/python tools/run_accuracy_ab.py <out_dir> \
        --per-scene 10 \
        --arms "base::|merge::MERGE_TEXT_SIM=0.65;MERGE_SAME_RISK_KEY=1|hires::SCENE_IMAGE_OVERRIDES=TEMPORARY_ELECTRICITY=1280:high;SCAFFOLD=1280:high"

Note the "::" between the arm name and its environment overrides, and "|"
between arms.

Arm syntax: name::ENV=VALUE;ENV=VALUE. Arms are separated by "|". The name
is mandatory so that values containing ":" never split an arm.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PY_BIN = str(BASE / ".venv" / "bin" / "python")
DEFAULT_SET = "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/评测数据/施工安全图集_140张"

# 基线配置：与当前发货配置一致，各臂在此之上只覆盖自己那几个变量。
BASELINE_ENV = {
    "REQUEST_TIMEOUT_SECONDS": "9",
    "DEEPSEEK_VISION_TIMEOUT_SECONDS": "8.4",
    "MODEL_MAX_ATTEMPTS": "3",
    "MODEL_RETRY_MIN_BUDGET": "1.0",
    "MODEL_CONNECT_TIMEOUT": "6.0",
    "IMAGE_DETAIL": "low",
    "MAX_IMAGE_EDGE": "768",
    "CHECKLIST_AUTO_MODE": "subject",
    "SCENE_CHECKLIST_SCOPE": "family_parts",
    "MAX_RISK_ITEMS": "3",
    "MAX_REPORT_ISSUES": "12",
    "ENUMERATE_PARTS": "0",
    "MAX_OBJECTS": "6",
}


def parse_arms(spec: str) -> list[tuple[str, dict[str, str]]]:
    arms: list[tuple[str, dict[str, str]]] = []
    for chunk in spec.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 名字与参数用 "::" 分隔：参数值里可能含冒号（如 1280:high）
        name, _, assignments = chunk.partition("::")
        overrides: dict[str, str] = {}
        for pair in assignments.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            key, _, value = pair.partition("=")
            overrides[key.strip()] = value.strip()
        arms.append((name.strip(), overrides))
    return arms


def pick(labels: list[dict], per_scene: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = {}
    for row in labels:
        by_scene.setdefault(row["test_scene"], []).append(row)
    picked: list[dict] = []
    for items in by_scene.values():
        if per_scene >= len(items):
            picked.extend(items)
            continue
        stride = len(items) / per_scene
        picked.extend(items[int(i * stride)] for i in range(per_scene))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir")
    parser.add_argument("--per-scene", type=int, default=10)
    parser.add_argument("--set", dest="dataset", default=DEFAULT_SET)
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument("--arms", default="base")
    parser.add_argument("--scenes", default="", help="只跑这些场景，逗号分隔；留空跑全部。")
    args = parser.parse_args()

    labels = [json.loads(line) for line in
              (Path(args.dataset) / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick(labels, args.per_scene)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        picked = [row for row in picked if row["test_scene"] in wanted]
    arms = parse_arms(args.arms)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"

    for index, label in enumerate(picked, start=1):
        image = Path(args.dataset) / label["image_path"]
        for name, overrides in arms:
            env = dict(os.environ)
            env.update(BASELINE_ENV)
            env.update(overrides)
            command = [PY_BIN, str(BASE / "src" / "safety_check.py"), str(image),
                       "--scene", label["test_scene"], "--branches", "4",
                       "--json-output", "--no-save"]
            started = time.perf_counter()
            proc = subprocess.run(command, cwd=BASE, env=env, capture_output=True, text=True)
            wall = time.perf_counter() - started
            try:
                payload = json.loads(proc.stdout[proc.stdout.index("{"):])
            except ValueError:
                payload = {"status": "error", "error": {"type": "unparsable",
                                                        "message": proc.stderr[-200:]}}
            usage = (payload.get("timings") or {}).get("token_usage") or {}
            record = {
                "sample_id": label["sample_id"],
                "version": name,
                "budget_seconds": float(env["REQUEST_TIMEOUT_SECONDS"]),
                "test_scene": label["test_scene"],
                "image_path": str(image),
                "wall_seconds": round(wall, 4),
                "branches": 4,
                "scope": env["SCENE_CHECKLIST_SCOPE"],
                "overrides": overrides,
                "result": {
                    "status": payload.get("status"),
                    "issues": payload.get("issues") or [],
                    "scene": payload.get("scene"),
                    "branch_status": payload.get("branch_status") or [],
                    "branch_messages": payload.get("branch_messages") or [],
                    "usage": usage,
                    "error": (payload.get("error") or {}).get("type") or "",
                },
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index:>3}/{len(picked)}] {name:<18} {label['sample_id']:<10} "
                  f"{wall:5.2f}s issues={len(record['result']['issues']):>2} "
                  f"{record['result']['error']}", file=sys.stderr, flush=True)
            if args.pace:
                time.sleep(args.pace)
    print(f"records: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
