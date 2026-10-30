#!/usr/bin/env python3
"""Import the main checker's runtime rule packs and their trigger catalogue.

The full checker ships 11 runtime rule packs with 1036 rules and 335
observable subjects (triggers), each subject carrying the cues that tell a
model how to recognise it. That is the catalogue the fast checker was missing:
the rule pack it had covered 63 rules and 62 subjects.

Nothing here reaches a prompt. The import produces a pool plus a coverage
report, and the prompt-side selection stays a local, capped lookup.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MAIN = Path("/Users/gjz/Downloads/safe-checker-v2-main 2/src/backend/resources/rules/generated")
OUT = BASE / "resources" / "main_system_rules.json"

# 主系统规则包文件 -> 我们用的场景键
SCENE_OF_PACK = {
    "gb50194": "TEMPORARY_ELECTRICITY",
    "temporary_electricity": "TEMPORARY_ELECTRICITY",
    "highwork": "WORK_AT_HEIGHT",
    "foundation_pit": "FOUNDATION_PIT",
    "lifting_operations": "LIFTING_OPERATIONS",
    "tower_crane": "TOWER_CRANE",
    "construction_hoist": "CONSTRUCTION_HOIST",
    "cantilever": "SCAFFOLD_CANTILEVER",
    "coupler_type": "SCAFFOLD_COUPLER_TYPE",
    "disc_buckle": "SCAFFOLD_DISC_BUCKLE",
    "attached_lifting": "SCAFFOLD_ATTACHED_LIFTING",
}
PACK_SCENE_HINT = {
    "gb50194": "施工用电",
    "temporary_electricity": "临时用电",
    "highwork": "高处作业",
    "foundation_pit": "基坑",
    "lifting_operations": "起重吊装",
    "tower_crane": "塔吊",
    "construction_hoist": "施工升降机",
    "cantilever": "悬挑式脚手架",
    "coupler_type": "扣件式脚手架",
    "disc_buckle": "盘扣式脚手架",
    "attached_lifting": "附着式升降脚手架",
}


def pack_key(path: Path) -> str:
    name = path.stem.replace("_runtime_rulepack", "").replace("_rulepack", "")
    for key in SCENE_OF_PACK:
        if name.startswith(key):
            return key
    return name


def main() -> int:
    if not MAIN.exists():
        print(f"找不到主系统规则包目录 {MAIN}", file=sys.stderr)
        return 1
    rules: list[dict] = []
    subjects: dict[str, dict] = {}
    by_pack: dict[str, int] = {}
    for path in sorted(MAIN.rglob("*.json")):
        key = pack_key(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rules") or []
        if not rows:
            continue
        by_pack[key] = len(rows)
        for row in rows:
            triggers = []
            for trigger in row.get("triggers") or []:
                trigger_id = str(trigger.get("trigger_id") or "").strip()
                if not trigger_id or trigger_id.endswith("-SCENE"):
                    continue
                triggers.append({
                    "trigger_id": trigger_id,
                    "trigger_name": str(trigger.get("trigger_name") or "").strip(),
                    "visibility_tag": str(trigger.get("visibility_tag") or "").strip(),
                    "recognition_mode": str(trigger.get("recognition_mode") or "").strip(),
                    "observable_cues": [str(c).strip() for c in trigger.get("observable_cues") or [] if str(c).strip()],
                })
                entry = subjects.setdefault(trigger_id, {
                    "trigger_id": trigger_id,
                    "subject": str(trigger.get("trigger_name") or "").strip(),
                    "visibility_tag": str(trigger.get("visibility_tag") or "").strip(),
                    "recognition_mode": str(trigger.get("recognition_mode") or "").strip(),
                    "observable_cues": [],
                    "scenes": [],
                    "rules": 0,
                })
                entry["rules"] += 1
                if entry["subject"] not in entry["scenes"]:
                    entry["scenes"].append(PACK_SCENE_HINT.get(key, key))
                for cue in trigger.get("observable_cues") or []:
                    cue = str(cue).strip()
                    if cue and cue not in entry["observable_cues"]:
                        entry["observable_cues"].append(cue)
            rules.append({
                "regulation_id": row.get("rulepack_id"),
                "spec_code": row.get("spec_code"),
                "spec_clause": row.get("spec_clause"),
                "clause_text": str(row.get("clause_text") or "").strip(),
                "judge_dimension": str(row.get("judge_dimension") or "").strip(),
                "judge_boundary": row.get("judge_boundary") or {},
                "applicability": row.get("applicability") or {},
                "evidence_requirements": row.get("evidence_requirements") or [],
                "disambiguation_notes": row.get("disambiguation_notes") or [],
                "keywords": row.get("keywords") or [],
                "similar_words": row.get("similar_words") or [],
                "disposal_suggestion": row.get("disposal_suggestion") or "",
                "scene": SCENE_OF_PACK.get(key, key),
                "scene_hint": PACK_SCENE_HINT.get(key, key),
                "triggers": triggers,
            })

    for entry in subjects.values():
        entry["scenes"] = sorted(set(entry["scenes"]))

    payload = {
        "source": str(MAIN),
        "note": "主系统运行时规则包导出的规则与可观测主体，仅作规则池与检索使用，不直接进提示词",
        "pack_rule_counts": by_pack,
        "rule_count": len(rules),
        "subject_count": len(subjects),
        "rules": rules,
        "subjects": sorted(subjects.values(), key=lambda x: x["trigger_id"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"导入规则 {len(rules)} 条，可观测主体 {len(subjects)} 个 -> {OUT}")
    for key, count in sorted(by_pack.items(), key=lambda x: -x[1]):
        print(f"  {PACK_SCENE_HINT.get(key, key):<12} {count:>4} 条")

    # 与我们自己的法规目录做条款覆盖对照
    try:
        from verified_regulations import VERIFIED_REGULATIONS, _norm_code, _norm_article
        ours = set()
        for record in VERIFIED_REGULATIONS.values():
            ours.add((_norm_code(record.get("standard_code")), _norm_article(record.get("article"))))
        theirs = set()
        for rule in rules:
            theirs.add((_norm_code(rule["spec_code"]), _norm_article(rule["spec_clause"])))
        print()
        print(f"主系统条款 {len(theirs)} 个；我们已核验目录 {len(ours)} 个")
        print(f"  主系统有、我们目录没有的条款: {len(theirs - ours)} 个")
        print(f"  我们目录有、主系统规则没用的条款: {len(ours - theirs)} 个")
    except Exception as exc:
        print(f"覆盖对照跳过: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
