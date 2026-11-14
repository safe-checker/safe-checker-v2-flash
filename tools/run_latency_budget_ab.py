#!/usr/bin/env python3
"""A/B the per-branch model budget against hit rate and tail latency.

P7 requires the warning delay to stay within 8 seconds. The detector runs
several branches concurrently and keeps whatever finished in time, so a
tighter per-branch budget guarantees the tail but drops the issues that the
slow branch would have found.

The two arms are run interleaved on the same images inside one time window,
because upstream latency drifts over hours and a cross-window comparison is
not a comparison.

Usage:
    .venv/bin/python tools/run_latency_budget_ab.py <out_dir> [--per-scene 10]
        [--arms "branch:8.4,total:9|branch:7.0,total:8"]
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


def parse_arm(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in spec.split(","):
        key, _, value = part.partition(":")
        out[key.strip()] = float(value)
    return out


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
    parser.add_argument("--arms", default="branch:8.4,total:9|branch:7.0,total:8")
    parser.add_argument("--tag", default="", help="给版本名加前缀，用来区分重复跑。")
    args = parser.parse_args()

    labels = [json.loads(line) for line in
              (Path(args.dataset) / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick(labels, args.per_scene)
    arms = [parse_arm(spec) for spec in args.arms.split("|")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"

    for index, label in enumerate(picked, start=1):
        image = Path(args.dataset) / label["image_path"]
        for arm in arms:
            name = f"{args.tag}b{arm['branch']:g}_t{arm['total']:g}"
            env = dict(
                os.environ,
                REQUEST_TIMEOUT_SECONDS=str(arm["total"]),
                DEEPSEEK_VISION_TIMEOUT_SECONDS=str(arm["branch"]),
                MODEL_MAX_ATTEMPTS="3",
                MODEL_RETRY_MIN_BUDGET="1.0",
                MODEL_CONNECT_TIMEOUT="6.0",
                IMAGE_DETAIL="low",
                MAX_IMAGE_EDGE="768",
                CHECKLIST_AUTO_MODE="subject",
                SCENE_CHECKLIST_SCOPE="family_parts",
                MAX_RISK_ITEMS="3",
                MAX_REPORT_ISSUES="12",
                ENUMERATE_PARTS="0",
                MAX_OBJECTS="6",
            )
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
                "budget_seconds": arm["total"],
                "branch_budget_seconds": arm["branch"],
                "test_scene": label["test_scene"],
                "image_path": str(image),
                "wall_seconds": round(wall, 4),
                "branches": 4,
                "scope": "family_parts",
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
            dropped = sum(1 for s in record["result"]["branch_status"] if s != "success")
            print(f"[{index:>3}/{len(picked)}] {name:<12} {label['sample_id']:<10} "
                  f"{wall:5.2f}s issues={len(record['result']['issues']):>2} "
                  f"丢弃分路={dropped} {record['result']['error']}",
                  file=sys.stderr, flush=True)
            if args.pace:
                time.sleep(args.pace)
    print(f"records: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
