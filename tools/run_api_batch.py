#!/usr/bin/env python3
"""Run a sample set through the live HTTP interface and record the results.

The CLI path and the service path differ in two ways that matter for
accuracy: the service truncates a slow branch instead of the whole image,
and it retries once when every branch timed out. Acceptance numbers should
therefore be taken on the service path, which is also what the video client
calls.

Usage:
    python tools/run_api_batch.py <out_dir> [--per-scene 10] [--base-url ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "acceptance_tests"))

from common import Inspector, load_manifest, write_jsonl  # noqa: E402

DEFAULT_SET = "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/评测数据/施工安全图集_140张"


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
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--pace", type=float, default=2.0)
    parser.add_argument("--version", default="api_parts")
    parser.add_argument("--scenes", default="", help="只跑这些场景，逗号分隔；留空跑全部。")
    args = parser.parse_args()

    labels = [json.loads(line) for line in
              (Path(args.dataset) / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = pick(labels, args.per_scene)
    if args.scenes:
        wanted = {s.strip() for s in args.scenes.split(",") if s.strip()}
        picked = [row for row in picked if row["test_scene"] in wanted]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    client = Inspector(base_url=args.base_url)
    for index, row in enumerate(picked, start=1):
        image = Path(args.dataset) / row["image_path"]
        status, payload, elapsed = client.inspect(image, scene=row["test_scene"])
        issues = (payload or {}).get("issues") or [] if isinstance(payload, dict) else []
        records.append({
            "sample_id": row["sample_id"],
            "version": args.version,
            "budget_seconds": 8.0,
            "test_scene": row["test_scene"],
            "image_path": str(image),
            "wall_seconds": round(elapsed, 4),
            "branches": (payload or {}).get("branches") if isinstance(payload, dict) else None,
            "scope": "family_parts",
            "result": {
                "status": (payload or {}).get("status") if isinstance(payload, dict) else None,
                "issues": issues,
                "scene": (payload or {}).get("scene") if isinstance(payload, dict) else None,
                "branch_status": (payload or {}).get("branch_status") or [] if isinstance(payload, dict) else [],
                "branch_messages": (payload or {}).get("branch_messages") or [] if isinstance(payload, dict) else [],
                "usage": ((payload or {}).get("timings") or {}).get("token_usage") or {} if isinstance(payload, dict) else {},
                "error": "" if status == 200 else f"HTTP {status}",
            },
        })
        print(f"[{index:>3}/{len(picked)}] {row['sample_id']:<10} {elapsed:5.2f}s "
              f"http={status} issues={len(issues):>2}", file=sys.stderr, flush=True)
        if args.pace:
            time.sleep(args.pace)
    client.close()
    write_jsonl(out_dir / "records.jsonl", records)
    print(f"records: {out_dir / 'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
