#!/usr/bin/env python3
"""P7 latency test: risk warning delay must not exceed 8 seconds.

The measurement is the round trip on the public interface, from the moment
the request leaves the client to the moment the response is parsed. That is
the interval the integration side actually waits for.

The outline asks for at least 1000 valid samples spread over 10 scenes with
at least 100 per scene. The script reports the shortfall instead of hiding
it, so a partial dataset cannot be mistaken for a compliant run.

Usage:
    python acceptance_tests/test_latency.py \
        --manifest acceptance_tests/dataset/manifest.jsonl \
        --out acceptance_tests/results/latency --pace 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from common import (LATENCY_THRESHOLD, SCENES, Inspector, load_manifest, print_table,
                    summarize, write_json, write_jsonl)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("acceptance_tests/dataset/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/results/latency"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--pace", type=float, default=2.0,
                        help="秒，图片之间的间隔。上游在持续压测下会变慢，节流能压低超时率。")
    parser.add_argument("--threshold", type=float, default=LATENCY_THRESHOLD,
                        help="P7 阈值（秒），默认 8。")
    parser.add_argument("--min-per-scene", type=int, default=100)
    parser.add_argument("--min-total", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=1,
                        help="网络瞬时故障时的重试次数，不用于掩盖真实超时。")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本，用于冒烟。")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]

    client = Inspector(base_url=args.base_url) if args.base_url else Inspector()
    records: list[dict[str, Any]] = []
    per_scene_valid: dict[str, int] = {code: 0 for code in SCENES}

    for index, sample in enumerate(manifest, start=1):
        code = sample.get("scene_code", "")
        attempt = 0
        status = None
        payload: Any = None
        elapsed = 0.0
        while True:
            try:
                status, payload, elapsed = client.inspect(
                    sample["image_path"], scene=sample.get("scene", ""))
                break
            except Exception as exc:  # 网络层失败，按重试次数处理
                attempt += 1
                elapsed = 0.0
                if attempt > args.retries:
                    payload = {"detail": f"{type(exc).__name__}: {exc}"}
                    break
                time.sleep(3.0)
        valid = status == 200 and isinstance(payload, dict) and payload.get("status") == "success"
        if valid:
            per_scene_valid[code] = per_scene_valid.get(code, 0) + 1
        records.append({
            "sample_id": sample.get("sample_id", ""),
            "scene_code": code,
            "scene": sample.get("scene", ""),
            "image_path": sample.get("image_path", ""),
            "source": sample.get("source", ""),
            "status_code": status,
            "valid": valid,
            "elapsed_seconds": round(elapsed, 4),
            "issue_count": (payload or {}).get("issue_count") if isinstance(payload, dict) else None,
            "error": None if valid else _error_of(payload),
            "attempts": attempt + 1,
        })
        print(f"[{index:>4}/{len(manifest)}] {code:<26} {elapsed:6.2f}s "
              f"{'OK' if valid else 'FAIL'} {records[-1]['error'] or ''}", flush=True)
        if args.pace:
            time.sleep(args.pace)

    valid_rows = [r for r in records if r["valid"]]
    values = [r["elapsed_seconds"] for r in valid_rows]
    over = [r for r in valid_rows if r["elapsed_seconds"] > args.threshold]
    scene_shortfall = {
        code: max(0, args.min_per_scene - per_scene_valid.get(code, 0))
        for code in SCENES
    }
    total_shortfall = max(0, args.min_total - len(valid_rows))

    report = {
        "metric": "P7 风险预警延迟",
        "threshold_seconds": args.threshold,
        "judgement": "从严口径：全部有效样本的延迟均不大于阈值",
        "planned_samples": len(manifest),
        "valid_samples": len(valid_rows),
        "invalid_samples": len(records) - len(valid_rows),
        "over_threshold_samples": len(over),
        "over_threshold_sample_ids": [r["sample_id"] for r in over][:50],
        "latency": summarize(values),
        "per_scene_valid": per_scene_valid,
        "sample_requirement": {
            "min_total": args.min_total,
            "min_per_scene": args.min_per_scene,
            "total_shortfall": total_shortfall,
            "per_scene_shortfall": {k: v for k, v in scene_shortfall.items() if v > 0},
        },
        "compliance": {
            "latency": len(over) == 0 and len(valid_rows) == len(records),
            "sample_sufficiency": total_shortfall == 0 and not any(scene_shortfall.values()),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "records.jsonl", records)
    write_json(args.out / "latency_report.json", report)

    print()
    print_table(["统计项", "数值"], [
        ["计划样本数", report["planned_samples"]],
        ["有效样本数", report["valid_samples"]],
        ["无效样本数", report["invalid_samples"]],
        ["最小值(s)", report["latency"].get("min", "-")],
        ["平均值(s)", report["latency"].get("mean", "-")],
        ["中位数(s)", report["latency"].get("median", "-")],
        ["P90(s)", report["latency"].get("p90", "-")],
        ["P95(s)", report["latency"].get("p95", "-")],
        ["最大值(s)", report["latency"].get("max", "-")],
        [f"超过 {args.threshold:g}s 的样本数", report["over_threshold_samples"]],
        ["样本量缺口", total_shortfall],
    ])
    print(f"\n延迟判定：{'符合' if report['compliance']['latency'] else '不符合'}"
          f"（阈值 {args.threshold:g}s，从严口径）")
    print(f"样本量判定：{'充足' if report['compliance']['sample_sufficiency'] else '不足'}"
          f"（缺 {total_shortfall} 条，场景缺口 {report['sample_requirement']['per_scene_shortfall']}）")
    print(f"记录：{args.out / 'records.jsonl'}")
    return 0 if report["compliance"]["latency"] else 1


def _error_of(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)[:160]
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message", ""))[:160]
    return str(payload)[:160]


if __name__ == "__main__":
    raise SystemExit(main())
