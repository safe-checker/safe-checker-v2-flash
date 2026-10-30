#!/usr/bin/env python3
"""Pick the rules that belong in one prompt.

The rule pool is meant to grow past the 63 hand-curated rules. Everything in it
cannot go into a prompt, so this picks the slice that matches the subjects the
image is being checked against. Purely local: no model call, no latency.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scene_checklists import SUBJECT_CHECKLIST

STOP = set("的了和与或及其之在有应不应须必须当对从为按照根据进行情况时地等上下前后内外")


def terms(text: str) -> set[str]:
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def build_subject_pool(rules_path: Path) -> list[dict]:
    """Group the rule pool by the subject it was bound to."""
    pool: dict[str, list[dict]] = {}
    if not rules_path.exists():
        return []
    for record in json.loads(rules_path.read_text(encoding="utf-8")):
        subject = record.get("subject") or ""
        if not subject:
            continue
        pool.setdefault(subject, []).append(record)
    return [{"subject": subject, "rules": items} for subject, items in pool.items()]


def score_rule(rule: dict, subject: dict) -> float:
    """How well a rule fits the subject it is being checked under."""
    subject_terms = terms(subject["subject"])
    for group in SUBJECT_CHECKLIST:
        if group["subject"] == subject["subject"]:
            subject_terms |= terms(" ".join(group.get("cues") or []))
            break
    rule_terms = terms(rule.get("judge_dimension", "")) | terms(
        (rule.get("judge_boundary") or {}).get("non_compliant", ""))
    if not rule_terms:
        return 0.0
    return len(rule_terms & subject_terms) / len(rule_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=str(BASE / "resources" / "generated_rules.json"))
    parser.add_argument("--subjects", default="", help="Comma separated subject names.")
    parser.add_argument("--top", type=int, default=6, help="Rules per subject.")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    pool = build_subject_pool(Path(args.pool))
    wanted = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not wanted:
        print("没有指定主体", file=sys.stderr)
        return 1

    lines = []
    total = 0
    for name in wanted:
        entry = next((p for p in pool if p["subject"] == name), None)
        if not entry:
            continue
        ranked = sorted(entry["rules"], key=lambda r: -score_rule(r, entry))[: args.top]
        for rule in ranked:
            boundary = rule.get("judge_boundary") or {}
            line = (f"- [{rule.get('standard_code','')} {rule.get('article','')}] "
                    f"{rule.get('judge_dimension','')}｜违规:{str(boundary.get('non_compliant',''))[:34]}")
            if total + len(line) > args.max_chars:
                break
            lines.append(line)
            total += len(line)
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[选了 {len(lines)} 条规则，{total} 字符]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
