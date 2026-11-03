#!/usr/bin/env python3
"""Run the detector over a set of images with several branches per image.

One process per image, and the branches run inside it over one HTTP
connection. Starting a process per branch used to mean four interpreter
starts, four catalogue loads and four TLS handshakes at the same time, and
the handshakes lost the race: the connect phase timed out and the image came
back empty.

Usage:
    .venv/bin/python tools/run_branches.py <out_dir> [images_per_scene] [--branches 2]

Environment:
    EVAL_SET             image set with the labels.jsonl layout
    EVAL_PACE_SECONDS    pause between images, the upstream slows down under load
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


def pick(labels: list[dict], per_scene: int) -> list[dict]:
    """Take an evenly spread sample of every scene.

    Taking the first N of a scene would take them from one sub folder only,
    so the sample would miss half of what that scene looks like.
    """
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
    parser = argparse.ArgumentParser(description="Run the detector over an image set.")
    parser.add_argument("out_dir")
    parser.add_argument("per_scene", nargs="?", type=int, default=10)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--set", dest="dataset", default=os.environ.get("EVAL_SET", DEFAULT_SET))
    parser.add_argument("--pace", type=float, default=float(os.environ.get("EVAL_PACE_SECONDS", "2")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("EVAL_TIMEOUT", "9")),
                        help="每张图的总预算（秒）。这一轮放宽到 9~10 秒换准确率。")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "records.jsonl"
    done: set[tuple[str, str]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["sample_id"], row["version"]))

    labels = [json.loads(line) for line in
              (dataset / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick(labels, args.per_scene)
    version = f"branches{args.branches}"
    print(f"取样 {len(picked)} 张，每张 {args.branches} 路，共 {len(picked) * args.branches} 次调用",
          file=sys.stderr)

    with path.open("a", encoding="utf-8") as fh:
        for index, label in enumerate(picked):
            if (label["sample_id"], version) in done:
                continue
            image = dataset / label["image_path"]
            env = dict(os.environ, REQUEST_TIMEOUT_SECONDS=str(args.timeout), MODEL_MAX_ATTEMPTS="3",
                       MODEL_RETRY_MIN_BUDGET="1.0", MODEL_CONNECT_TIMEOUT="6.0",
                       IMAGE_DETAIL="low", MAX_IMAGE_EDGE="768",
                       CHECKLIST_AUTO_MODE="subject", SCENE_CHECKLIST_SCOPE="family")
            command = [PY_BIN, str(BASE / "src" / "safety_check.py"), str(image),
                       "--scene", label["test_scene"], "--branches", str(args.branches),
                       "--json-output", "--no-save"]
            started = time.perf_counter()
            proc = subprocess.run(command, cwd=BASE, env=env, capture_output=True, text=True)
            wall = time.perf_counter() - started
            try:
                payload = json.loads(proc.stdout[proc.stdout.index("{"):])
            except ValueError:
                payload = {"status": "error", "error": {"type": "unparsable",
                                                         "message": proc.stderr[-200:]}}
            # 整张图都没有结果时，隔一会儿再试一次。
            # 上游会周期性地变慢（实测连续几十分钟超时率从 8% 涨到 50% 再回落），
            # 这种窗口里一次失败不代表这张图有问题。
            retries = 0
            while not payload.get("issues") and retries < 2:
                retries += 1
                time.sleep(6.0)
                proc = subprocess.run(command, cwd=BASE, env=env, capture_output=True, text=True)
                try:
                    payload = json.loads(proc.stdout[proc.stdout.index("{"):])
                except ValueError:
                    payload = {"status": "error", "error": {"type": "unparsable",
                                                             "message": proc.stderr[-200:]}}
            usage = (payload.get("timings") or {}).get("token_usage") or {}
            record = {
                "sample_id": label["sample_id"],
                "version": version,
                "budget_seconds": args.timeout,
                "test_scene": label["test_scene"],
                "image_path": str(image),
                "wall_seconds": round(wall, 4),
                "branches": args.branches,
                "retries": retries,
                "result": {
                    "status": payload.get("status"),
                    "issues": payload.get("issues") or [],
                    "scene": payload.get("scene"),
                    "model_detected_scene": payload.get("model_detected_scene"),
                    "branch_status": payload.get("branch_status") or [],
                    "branch_messages": payload.get("branch_messages") or [],
                    "usage": usage,
                    "error": (payload.get("error") or {}).get("type") or "",
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{index + 1:>3}/{len(picked)}] {label['sample_id']:<10} {wall:5.2f}s "
                  f"issues={len(record['result']['issues']):>2} "
                  f"{record['result']['branch_status']} {record['result']['error']}",
                  file=sys.stderr, flush=True)
            if args.pace:
                time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
