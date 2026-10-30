#!/usr/bin/env python3
"""Gate the generated rules before any of them may reach a prompt.

A rule that reaches the checker has to earn its place. Four gates:

1. **Clause alignment** - the judgement has to be traceable to the clause it
   came from. If the dimension names objects that never appear in the clause
   text, the model invented it.
2. **Subject binding** - the rule has to attach to one of the observable
   subjects. The full checker does the same thing: a rule without a subject is
   not runnable, because nothing in a photograph can confirm it.
3. **Loose boundaries** - a boundary that says "usually" or "in general" cannot
   be used to decide CLEAR. Loose boundaries turn into false positives.
4. **Family** - only rules about something visible may drive a check. Rules
   about paperwork, design calculation or tests stay in the citation layer.
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

# 边界里出现这些词就说明判定不干脆
LOOSE = re.compile(r"疑似|可能|一般情况|通常|适当|宜|必要时|视情况|原则上|大多|多数情况")
# 只在条文里出现的词不算依据
STOP = set("的了和与或及其之在有应不应须必须当对从为按照根据进行情况时地等上下前后内外")
# 太泛的名字不能当主体，否则清单里会出现"脚手架"这种无处可指的东西
GENERIC_SUBJECT = re.compile(r"^(?:现场|画面|图片|施工|作业|工程|部位|区域|其他|其|该|本)")


def terms(text: str) -> set[str]:
    """Two-character Chinese chunks, used as a cheap proxy for content words."""
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


# 维度里"不判断…"那段是模型自己声明的边界，写得越清楚越不该被当成"没依据"
EXCLUSION_TAIL = re.compile(r"[，,；;]\s*(不判断|不得判断|不涉及|不包括|不评价|不裁决).*$")


def core_dimension(dimension: str) -> str:
    """The part of a dimension that states what is actually judged."""
    return EXCLUSION_TAIL.split(str(dimension or ""))[0].strip() or str(dimension or "")


def subject_index() -> list[tuple[str, set[str]]]:
    index = []
    for group in SUBJECT_CHECKLIST:
        name = group["subject"]
        cues = " ".join(group.get("cues") or [])
        index.append((name, terms(name) | terms(cues)))
    return index


def align_score(dimension: str, boundary: str, clause_text: str) -> float:
    """How much of the clause the rule actually reflects.

    The direction matters. A precise dimension is written in the model's own
    words and says what it does *not* judge, so asking "how many of the rule's
    words appear in the clause" punishes exactly the rules we want. The
    question that catches invented rules is the opposite one: does the rule
    cover the content of the clause it came from?
    """
    clause_terms = {t for t in terms(clause_text) if t not in STOP and not t.isdigit()}
    if not clause_terms:
        return 0.0
    rule_terms = terms(core_dimension(dimension)) | terms(boundary)
    return len(clause_terms & rule_terms) / len(clause_terms)


def bind_subject(hint: str, dimension: str, index) -> tuple[str, float]:
    hint_terms = terms(hint)
    dim_terms = terms(dimension)
    best, best_score = "", 0.0
    for name, subject_terms in index:
        score = len(hint_terms & subject_terms) / max(len(hint_terms), 1) if hint_terms else 0.0
        if not score:
            score = len(dim_terms & subject_terms) / max(len(dim_terms), 1) * 0.5
        if score > best_score:
            best, best_score = name, score
    return (best, round(best_score, 3)) if best_score >= 0.34 else ("", round(best_score, 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="src", default=str(BASE / "resources" / "generated_rules_raw.json"))
    parser.add_argument("--out", default=str(BASE / "resources" / "generated_rules.json"))
    parser.add_argument("--report", default=str(BASE / "resources" / "generated_rules_report.json"))
    args = parser.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    clauses = {row["regulation_id"]: row for row in
               json.loads((BASE / "resources" / "standards" / "cleaned_clauses.json").read_text(encoding="utf-8"))}
    index = subject_index()

    accepted, rejected = [], []
    counts = {"align": 0, "loose": 0, "subject": 0, "family": 0, "shape": 0}
    for clause in raw:
        clause_text = str((clauses.get(clause.get("regulation_id")) or {}).get("clause_text") or "")
        for rule in clause.get("rules") or []:
            dimension = str(rule.get("judge_dimension") or "").strip()
            boundary = rule.get("judge_boundary") or {}
            non_compliant = str(boundary.get("non_compliant") or "")
            compliant = str(boundary.get("compliant") or "")
            doubtful = str(boundary.get("doubtful") or "")
            family = str(rule.get("rule_family") or "").strip()

            def reject(reason: str) -> None:
                counts[reason] += 1
                rejected.append({"atomic_rule_id": rule.get("atomic_rule_id"),
                                 "regulation_id": clause.get("regulation_id"),
                                 "article": clause.get("article"), "reason": reason,
                                 "judge_dimension": dimension})

            if not dimension or not non_compliant or not compliant or not doubtful:
                reject("shape")
                continue
            if family in {"ledger_only", "system_topology"}:
                reject("family")
                continue
            if LOOSE.search(non_compliant) or LOOSE.search(compliant):
                reject("loose")
                continue
            score = align_score(dimension, non_compliant, clause_text)
            if score < 0.25:
                reject("align")
                continue
            subject, subject_score = bind_subject(str(rule.get("subject_hint") or ""), dimension, index)
            origin = "existing"
            cue = ""
            if not subject:
                # 模型给出的主体名如果具体、可定位，就当作新主体收下。
                # 主系统也是这么做的：trigger 就是主体，规则包里没有的可以新增。
                # 风险是"主体其实不存在"，但那只会让模型找不到它、少查一项，
                # 不会变成误报，所以这个方向是安全的。
                hint = str(rule.get("subject_hint") or "").strip()
                if len(hint) < 4 or GENERIC_SUBJECT.search(hint):
                    reject("subject")
                    continue
                subject = hint
                origin = "new"
                cue = str(rule.get("subject_cue") or "").strip()
            record = dict(rule)
            record.update({
                "regulation_id": clause.get("regulation_id"),
                "standard_code": clause.get("standard_code"),
                "article": clause.get("article"),
                "clause_text": clause_text,
                "align_score": round(score, 3),
                "subject": subject,
                "subject_score": subject_score,
                "subject_origin": origin,
                "subject_cue": cue,
                "scene_hint": str(rule.get("scene_hint") or "GENERAL_SITE"),
                "runtime_enabled": True,
                "risk_key": str(rule.get("atomic_rule_id") or "").split("@")[-1].upper(),
            })
            accepted.append(record)

    Path(args.out).write_text(json.dumps(accepted, ensure_ascii=False, indent=1), encoding="utf-8")
    by_std = {}
    for record in accepted:
        key = record.get("standard_code")
        by_std[key] = by_std.get(key, 0) + 1
    report = {
        "input_clauses": len(raw),
        "generated_rules": sum(len(c.get("rules") or []) for c in raw),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reject_reasons": counts,
        "accepted_by_standard": by_std,
        "rejected_detail": rejected[:400],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"生成 {report['generated_rules']} 条 -> 通过 {len(accepted)} 条，剔除 {len(rejected)} 条")
    print(f"剔除原因: {counts}")
    print(f"按标准: {by_std}")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
