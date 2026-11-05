#!/usr/bin/env python3
"""Score an A/B run from LLM-judged verdicts.

Usage:
    .venv/bin/python tools/score_arms.py <result_dir> [verdicts.json]

The verdicts must come from tools/judge_detections.py. A lexical comparison
would score every paraphrase as a miss.

Records the judge could not answer are left out of the denominator and
reported separately: a judge failure is not a detection failure, and counting
it as a miss drags every number down.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/safety_eval/results/final_merged")
    verdict_file = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "judge_v2.json"
    if not verdict_file.exists():
        verdict_file = src / "judge_new.json"
    recs = [json.loads(line) for line in
            (src / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    verd = json.loads(verdict_file.read_text(encoding="utf-8"))
    judged = {(row["sample_id"], row["version"]): row["judge"] for row in verd}

    print(f"记录 {len(recs)} 条，裁判结果 {verdict_file.name}")
    arms = sorted({row["version"] for row in recs})
    print(f"{'臂':<30}{'命中/已判':>14}{'有输出':>12}{'空输出':>8}{'判不了':>8}"
          f"{'中位耗时':>10}{'p90':>8}{'max':>8}")
    for arm in arms:
        rows = [row for row in recs if row["version"] == arm]
        scored = [row for row in rows if (row["sample_id"], arm) in judged
                  and not judged[(row["sample_id"], arm)].get("_unjudged")]
        unjudged = len(rows) - len(scored)
        hits = sum(1 for row in scored if judged[(row["sample_id"], arm)].get("hit"))
        ok = [row for row in rows if row["result"].get("issues")]
        times = sorted(row["wall_seconds"] for row in rows)
        rate = f"{hits}/{len(scored)} = {hits/max(len(scored),1)*100:.1f}%"
        print(f"{arm:<30}{rate:>14}{len(ok):>12}{len(rows)-len(ok):>8}{unjudged:>8}"
              f"{statistics.median(times):>10.2f}{times[int(len(times)*0.9)-1]:>8.2f}{max(times):>8.2f}")

    if len(arms) > 1:
        ids = sorted({row["sample_id"] for row in recs})
        common = [sid for sid in ids
                  if all((sid, arm) in judged and not judged[(sid, arm)].get("_unjudged")
                         for arm in arms)]
        if common:
            print(f"\n配对（同 {len(common)} 张图全部判得了）:")
            for arm in arms:
                hits = sum(1 for sid in common if judged[(sid, arm)].get("hit"))
                print(f"   {arm:<30}{hits}/{len(common)} = {hits/len(common)*100:.1f}%")
            base = arms[0]
            for arm in arms[1:]:
                only_new = sum(1 for sid in common
                               if judged[(sid, arm)].get("hit") and not judged[(sid, base)].get("hit"))
                only_base = sum(1 for sid in common
                                if judged[(sid, base)].get("hit") and not judged[(sid, arm)].get("hit"))
                print(f"   {arm} 独赢 {only_new} 张，{base} 独赢 {only_base} 张 → 净 {only_new-only_base:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
