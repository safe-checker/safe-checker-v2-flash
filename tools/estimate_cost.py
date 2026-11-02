#!/usr/bin/env python3
"""Estimate what a run costs, and what the cache is saving.

DeepSeek bills the uncached part of the prompt at the full rate and the cached
part at a much lower one. The system prompt is identical for every image of the
same scene and branch, so most of it should be served from the cache. This
script reads the token usage out of a run and prices it.

Usage:
    .venv/bin/python tools/estimate_cost.py <result_dir> [--rate-in 2 --rate-out 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate the cost of a run.")
    parser.add_argument("result_dir")
    parser.add_argument("--rate-in", type=float, default=2.0,
                        help="每百万输入 token 的价格，未命中缓存的部分")
    parser.add_argument("--rate-out", type=float, default=4.0,
                        help="每百万输出 token 的价格")
    parser.add_argument("--rate-cache", type=float, default=0.2,
                        help="每百万命中缓存的输入 token 的价格")
    args = parser.parse_args()

    path = Path(args.result_dir) / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = completion = hit = miss = 0
    calls = 0
    for row in rows:
        usage = (row.get("result") or {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        hit += int(usage.get("cache_hit_tokens") or 0)
        miss += int(usage.get("cache_miss_tokens") or 0)
        calls += len((row.get("result") or {}).get("branch_status") or [])
    if not prompt:
        print("这批记录里没有 token 用量（旧批次没存），无法估算")
        return 1
    cost_in = (hit / 1e6) * args.rate_cache + (miss / 1e6) * args.rate_in
    cost_out = (completion / 1e6) * args.rate_out
    total = cost_in + cost_out
    print(f"记录 {len(rows)} 条，视觉调用 {calls} 次")
    print(f"输入 {prompt/1e6:.2f}M token（缓存命中 {hit/1e6:.2f}M / 未命中 {miss/1e6:.2f}M，"
          f"命中率 {hit/max(prompt,1)*100:.1f}%）")
    print(f"输出 {completion/1e6:.2f}M token")
    print(f"估算费用：输入 ¥{cost_in:.2f} + 输出 ¥{cost_out:.2f} = ¥{total:.2f}")
    print(f"单次视觉调用约 ¥{total/max(calls,1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
