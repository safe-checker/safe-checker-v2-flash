#!/usr/bin/env python3
"""Generate per-scene inspection checklists from the rule pack.

One line per unique check dimension. Duplicate rules that cover the same
dimension (the personal protective equipment rules appear in several scene
rule packs) are merged so the checklist stays short enough for the prompt.

Two consistency rules are applied while rendering:

* dimensions that the existing object-and-state protocol already governs are
  rendered as a pointer to that protocol instead of repeating a looser
  boundary, so the checklist can never weaken the stricter protocol;
* text is shortened at clause boundaries, never mid-phrase.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE))
from catalogs.rulepack_catalog import RULEPACK_ROWS  # noqa: E402

TARGET = BASE / "src" / "catalogs" / "scene_checklists.py"

# 全部场景铺开。规则包覆盖的每个场景都注入对应清单。
ENABLED_SCENES = [
    "SCAFFOLD_CANTILEVER",
    "SCAFFOLD_COUPLER_TYPE",
    "SCAFFOLD_DISC_BUCKLE",
    "SCAFFOLD_ATTACHED_LIFTING",
    "FOUNDATION_PIT",
    "TOWER_CRANE",
    "CONSTRUCTION_HOIST",
    "LIFTING_OPERATIONS",
    "TEMPORARY_ELECTRICITY",
    "WORK_AT_HEIGHT",
    "GENERAL_SITE",
]

# AUTO 场景下，每个已放开场景给出的核心维度数量
AUTO_CORE_DIMENSIONS = 3

# 清单行是否渲染主体锚点（[主体名] 前缀）。
# 94 张配对实测：开 76/93（81.7%），关 79/94（84.0%），方向为负且没有收益，
# 所以默认关闭。设 CHECKLIST_SUBJECT_ANCHOR=1 可以再打开做对照。
SUBJECT_ANCHOR_ENABLED = os.environ.get("CHECKLIST_SUBJECT_ANCHOR", "0").strip() == "1"

# 场景家族。模型把子类判错的概率不低，但把"脚手架"这种家族判对的比例很高，
# 所以 AUTO 路径按家族给清单，子类特有维度标注适用条件，由模型自行取舍。
SCENE_FAMILIES: dict[str, dict] = {
    "SCAFFOLD_FAMILY": {
        "label": "脚手架",
        "scenes": [
            "SCAFFOLD_CANTILEVER",
            "SCAFFOLD_COUPLER_TYPE",
            "SCAFFOLD_DISC_BUCKLE",
            "SCAFFOLD_ATTACHED_LIFTING",
        ],
        "aliases": ["脚手架", "scaffold", "脚手架家族"],
    },
    "LIFTING_FAMILY": {
        "label": "起重与升降设备",
        "scenes": ["TOWER_CRANE", "CONSTRUCTION_HOIST", "LIFTING_OPERATIONS"],
        "aliases": ["起重设备", "升降设备", "lifting_family"],
    },
    "TEMPORARY_ELECTRICITY": {
        "label": "临时用电",
        "scenes": ["TEMPORARY_ELECTRICITY"],
        "aliases": ["临时用电", "配电箱", "electricity"],
    },
    "FOUNDATION_PIT": {
        "label": "基坑工程",
        "scenes": ["FOUNDATION_PIT"],
        "aliases": ["基坑", "foundation_pit"],
    },
    "WORK_AT_HEIGHT": {
        "label": "高处作业",
        "scenes": ["WORK_AT_HEIGHT"],
        "aliases": ["高处作业", "work_at_height"],
    },
    "GENERAL_SITE": {
        "label": "施工现场综合",
        "scenes": ["GENERAL_SITE"],
        "aliases": ["综合", "general_site"],
    },
}

# AUTO 路径每个家族给出的核心维度数量
FAMILY_CORE_DIMENSIONS = 5

# 出现在所有场景的通用劳保维度。人物优先检查协议已经逐部位覆盖，
# 清单里再列一遍只会挤掉能区分场景的维度。
# 顺序固定：写成 set 会在生成文件里留下随进程变化的顺序，仓库每次重生成都有 diff
GLOBAL_DIMENSIONS = (
    "图片中所有人员是否佩戴安全帽",
    "图片中所有人员双脚是否穿工作鞋",
    "图片中所有人员是否穿工作服",
    "图片中所有人员是否佩戴安全带",
)

SCENE_SHORT_LABELS = {
    "SCAFFOLD_CANTILEVER": "悬挑式",
    "SCAFFOLD_COUPLER_TYPE": "扣件式",
    "SCAFFOLD_DISC_BUCKLE": "盘扣式",
    "SCAFFOLD_ATTACHED_LIFTING": "附着升降式",
    "TOWER_CRANE": "塔式起重机",
    "CONSTRUCTION_HOIST": "施工升降机",
    "LIFTING_OPERATIONS": "起重吊装",
    "TEMPORARY_ELECTRICITY": "临时用电",
    "FOUNDATION_PIT": "基坑",
    "WORK_AT_HEIGHT": "高处作业",
    "GENERAL_SITE": "综合",
}

# 规则包原文里的个别笔误，渲染时修正并保留记录
TYPO_FIXES = {
    "已设置道中断": "已设置处中断",
    # 规则包里“条款类型”标记不应出现在给模型看的判定边界里
    "不满足所引推荐条款": "即不满足本项要求",
    "不满足所引强制条款": "即不满足本项要求",
}

# 这些维度由现有「对象与状态协议」管辖，清单只做指向，避免与协议口径冲突
PROTOCOL_REFERENCES = [
    (("积水", "地面积水"), "对象与状态协议第2条"),
    (("栏杆全封闭", "封闭防护", "安全防护设施的可见损坏", "密目安全网或小眼网"),
     "对象与状态协议第3条"),
]


ANCHOR_RE = re.compile(r"^(- )\[[^\]]{1,20}\] ")


def strip_anchor(line: str) -> str:
    """Remove the subject anchor prefix from one checklist line."""
    return ANCHOR_RE.sub(r"\1", line)


def subject_anchor(rows: list[dict]) -> str:
    """The observable subject a dimension is judged on.

    The rule pack binds every rule to one or more triggers, and a trigger
    names a subject that can actually be located in the photo together with the
    cues for recognising it. Rendering the subject in front of a checklist line
    tells the model where to look; the dimension text alone never says it.
    """
    counts: dict[str, int] = {}
    for row in rows:
        for trigger in row.get("triggers") or []:
            name = str(trigger.get("trigger_name") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    best = max(counts.items(), key=lambda item: (item[1], -len(item[0])))
    return best[0]


def merge_boundaries(rows: list[dict]) -> dict[str, str]:
    out = {"compliant": "", "non_compliant": "", "doubtful": ""}
    for row in rows:
        boundary = row.get("judge_boundary") or {}
        for key in out:
            value = str(boundary.get(key) or "").strip()
            if value and value not in out[key]:
                out[key] = value if not out[key] else out[key]
    return out


def shorten(text: str, limit: int) -> str:
    """Shorten at a clause boundary so the model never reads a dangling phrase."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    for wrong, right in TYPO_FIXES.items():
        text = text.replace(wrong, right)
    if len(text) <= limit:
        return text.rstrip("；;，,、")
    window = text[: limit + 1]
    cut = max(window.rfind(mark) for mark in "，,；;。、")
    if cut >= limit // 3:
        return text[:cut].rstrip("；;，,、")
    # 没有合适的分句点时，切在并列连接词之前，避免留下半个短语
    for mark in ("或", "及", "和", "与", "的"):
        pos = window.rfind(mark)
        if pos >= limit // 2:
            return text[:pos].rstrip("；;，,、")
    return text[:limit].rstrip("；;，,、") + "…"


def protocol_for(dimension: str) -> str | None:
    for keywords, reference in PROTOCOL_REFERENCES:
        if any(keyword in dimension for keyword in keywords):
            return reference
    return None


DIMENSION_PREFIX_RE = re.compile(r"^(?:推荐条款|强制条款|参考条款)\s*[：:]\s*")


def render_line(dimension: str, boundary: dict[str, str], anchor: str = "", scope: str = "") -> str:
    dimension = re.sub(r"\s+", " ", str(dimension)).strip()
    dimension = DIMENSION_PREFIX_RE.sub("", dimension)
    dimension = dimension.rstrip("。．.")
    dimension = dimension.replace("图片中所有人员双脚是否", "人员双脚是否")
    dimension = dimension.replace("图片中所有人员是否", "人员是否")
    label = dimension
    if scope:
        label = f"{dimension}（{scope}）"
    if anchor:
        label = f"[{anchor}] {label}"
    reference = protocol_for(dimension)
    if reference:
        return f"- {label}｜判定：按「{reference}」"
    return "｜".join([
        f"- {label}",
        f"违规:{shorten(boundary.get('non_compliant', ''), 32)}",
        f"合规:{shorten(boundary.get('compliant', ''), 28)}",
        f"存疑:{shorten(boundary.get('doubtful', ''), 28)}",
    ])


def dimension_scene_counts() -> dict[str, int]:
    """How many scenes each check dimension appears in.

    A dimension that only exists for one scene is what tells the model apart
    from the general safety checks, so it must come first in the checklist.
    """
    counts: dict[str, int] = {}
    for row in RULEPACK_ROWS:
        dimension = str(row.get("judge_dimension") or "").strip()
        if not dimension:
            continue
        for scene in row["scene_types"]:
            counts.setdefault(dimension, set())  # type: ignore[arg-type]
    for row in RULEPACK_ROWS:
        dimension = str(row.get("judge_dimension") or "").strip()
        if dimension:
            counts[dimension] = len({s for r in RULEPACK_ROWS
                                      if str(r.get("judge_dimension") or "").strip() == dimension
                                      for s in r["scene_types"]})
    return {k: v for k, v in counts.items() if isinstance(v, int)}


def group_by_dimension(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("judge_dimension") or "").strip(), []).append(row)
    return {dim: group for dim, group in grouped.items() if dim}


def build() -> dict[str, dict]:
    result: dict[str, dict] = {}
    scenes = sorted({scene for row in RULEPACK_ROWS for scene in row["scene_types"]})
    spread = dimension_scene_counts()
    for scene in scenes:
        rows = [row for row in RULEPACK_ROWS if scene in row["scene_types"]]
        grouped = group_by_dimension(rows)
        rendered = [
            (dim, render_line(dim, merge_boundaries(group), subject_anchor(group)))
            for dim, group in grouped.items()
        ]
        # 场景专属性强的排前面：只在本场景出现的维度优先于通用劳保维度
        rendered.sort(key=lambda item: (spread.get(item[0], 99), item[0]))
        lines = [line for _, line in rendered]
        result[scene] = {
            "scene": scene,
            "rule_count": len(rows),
            "dimension_count": len(lines),
            "lines": [strip_anchor(line) for line in lines],
            "lines_with_anchor": lines,
            "enabled": scene in ENABLED_SCENES,
        }
    return result


def build_families(scene_data: dict[str, dict]) -> dict[str, dict]:
    """Merge the scenes of a family into one checklist.

    The model recognises a family far more reliably than a subtype, so the
    merged checklist is the union of the member scenes. A dimension that only
    exists in some subtypes is marked with the subtypes it applies to, so the
    model can skip it when the subtype is unknown instead of reporting a risk
    for a structure that is not in the photo.
    """
    families: dict[str, dict] = {}
    for family, spec in SCENE_FAMILIES.items():
        members = [scene for scene in spec["scenes"] if scene in scene_data]
        if not members:
            continue
        rows = [row for row in RULEPACK_ROWS if any(scene in row["scene_types"] for scene in members)]
        grouped = group_by_dimension(rows)
        # 每个维度覆盖了家族内哪些子类
        covered: dict[str, list[str]] = {}
        for dim in grouped:
            covered[dim] = [
                scene for scene in members
                if any(dim == str(row.get("judge_dimension") or "").strip()
                       for row in RULEPACK_ROWS if scene in row["scene_types"])
            ]
        rendered = []
        for dim, group in grouped.items():
            subset = covered[dim]
            scope = ""
            if len(subset) < len(members):
                names = "、".join(SCENE_SHORT_LABELS.get(scene, scene) for scene in subset)
                scope = f"仅{names}适用"
            rendered.append((dim, render_line(dim, merge_boundaries(group), subject_anchor(group), scope)))
        # 家族共有的维度排前面，子类特有维度排后面
        rendered.sort(key=lambda item: (-len(covered[item[0]]), item[0]))
        lines = [line for _, line in rendered]
        # 核心维度跳过全局 PPE 维度：人物优先检查协议已经逐部位覆盖，
        # 家族清单的名额要留给能区分场景的设备与结构维度。
        core = [(dim, line) for dim, line in rendered if dim not in GLOBAL_DIMENSIONS]
        families[family] = {
            "family": family,
            "label": spec["label"],
            "scenes": members,
            "scene_labels": [SCENE_SHORT_LABELS.get(scene, scene) for scene in members],
            "rule_count": len(rows),
            "dimension_count": len(lines),
            "lines": [strip_anchor(line) for line in lines],
            "lines_with_anchor": lines,
            "core_lines": [strip_anchor(line) for _, line in core],
            "core_lines_with_anchor": [line for _, line in core],
        }
    return families


def family_core_lines(family_data: dict[str, dict], limit: int) -> list[str]:
    """Preview lines for the AUTO path, family by family."""
    out: list[str] = []
    for family, data in family_data.items():
        lines = data["lines"]
        out.append(f"【{data['label']}】核心维度{len(lines[:limit])}项：")
        out.extend(lines[:limit])
    return out


SCENE_FAMILY_BY_SCENE = {
    "SCAFFOLD_CANTILEVER": ["SCAFFOLD"], "SCAFFOLD_COUPLER_TYPE": ["SCAFFOLD"],
    "SCAFFOLD_DISC_BUCKLE": ["SCAFFOLD"], "SCAFFOLD_ATTACHED_LIFTING": ["SCAFFOLD"],
    "TOWER_CRANE": ["LIFTING"], "CONSTRUCTION_HOIST": ["LIFTING"], "LIFTING_OPERATIONS": ["LIFTING"],
    "TEMPORARY_ELECTRICITY": ["ELECTRICITY"], "FOUNDATION_PIT": ["PIT"],
    "WORK_AT_HEIGHT": ["HEIGHT"], "GENERAL_SITE": ["GENERAL"],
}

SCENE_FAMILY = {
    "SCAFFOLD_CANTILEVER": "SCAFFOLD", "SCAFFOLD_COUPLER_TYPE": "SCAFFOLD",
    "SCAFFOLD_DISC_BUCKLE": "SCAFFOLD", "SCAFFOLD_ATTACHED_LIFTING": "SCAFFOLD",
    "TOWER_CRANE": "LIFTING", "CONSTRUCTION_HOIST": "LIFTING", "LIFTING_OPERATIONS": "LIFTING",
    "TEMPORARY_ELECTRICITY": "ELECTRICITY",
    "FOUNDATION_PIT": "PIT", "WORK_AT_HEIGHT": "HEIGHT", "GENERAL_SITE": "GENERAL",
}


def scene_family(scene: str) -> str:
    return SCENE_FAMILY.get(scene, "GENERAL")


def scene_families(scene: str) -> list[str]:
    """The device families a scene belongs to."""
    return SCENE_FAMILY_BY_SCENE.get(scene, [])


def build_subjects() -> list[dict]:
    """Group the rules by the subject they are judged on.

    This is the single-call form of the trigger-to-rule mechanism: the rule
    pack binds every rule to one or more triggers, and a trigger names a
    subject that can be located in the photo. Listing the subjects first lets
    the model decide which subjects are actually present and check only the
    rules bound to them, instead of having to name the scene first.
    """
    # 同一个主体名可能挂在不同 trigger_id 上（劳保类就有四个），
    # 渲染给模型看的是主体名，所以按主体名合并，不按 trigger_id 拆。
    groups: dict[str, dict] = {}
    for row in RULEPACK_ROWS:
        dimension = str(row.get("judge_dimension") or "").strip()
        if not dimension:
            continue
        for trigger in row.get("triggers") or []:
            name = str(trigger.get("trigger_name") or "").strip()
            if not name:
                continue
            key = name
            group = groups.setdefault(key, {
                "trigger_id": str(trigger.get("trigger_id") or name),
                "subject": name,
                "cues": [str(c).strip() for c in trigger.get("observable_cues") or [] if str(c).strip()],
                "dimensions": [],
                "scenes": set(),
            })
            if not any(d["dimension"] == dimension for d in group["dimensions"]):
                group["dimensions"].append({
                    "dimension": dimension,
                    "boundary": row.get("judge_boundary") or {},
                    "scenes": sorted(set(row.get("scene_types") or [])),
                    "source": "rulepack",
                })
            group["scenes"].update(row.get("scene_types") or [])
    subjects = []
    for group in groups.values():
        group["scenes"] = sorted(group["scenes"])
        subjects.append(group)
    # 主体归属的规则包前缀决定它属于哪一族，供并发分组使用
    for group in subjects:
        group["families"] = sorted({scene_family(scene) for scene in group["scenes"]})
    # 通用人员主体排前面，人员防护是每张图都要看的
    subjects.sort(key=lambda item: (0 if item["subject"] == "图片中人员" else 1, -len(item["dimensions"]), item["subject"]))
    subjects.extend(_generated_subjects())
    return subjects


GENERATED_RULES_PATH = BASE / "data" / "generated" / "rules.json"
# 场景键 -> 设备族
SCENE_HINT_FAMILY = {
    "SCAFFOLD_CANTILEVER": "SCAFFOLD", "SCAFFOLD_COUPLER_TYPE": "SCAFFOLD",
    "SCAFFOLD_DISC_BUCKLE": "SCAFFOLD", "SCAFFOLD_ATTACHED_LIFTING": "SCAFFOLD",
    "SCAFFOLD_FAMILY": "SCAFFOLD",
    "TOWER_CRANE": "LIFTING", "CONSTRUCTION_HOIST": "LIFTING",
    "LIFTING_OPERATIONS": "LIFTING", "LIFTING_FAMILY": "LIFTING",
    "TEMPORARY_ELECTRICITY": "ELECTRICITY", "FOUNDATION_PIT": "PIT",
    "WORK_AT_HEIGHT": "HEIGHT", "GENERAL_SITE": "GENERAL",
}


def stable_id(name: str) -> str:
    """A trigger id that does not change between runs.

    Python randomises string hashing per process, so hash() would give every
    rebuild a different id and the repository would show a diff each time.
    """
    import hashlib

    return hashlib.md5(name.encode("utf-8")).hexdigest()[:8].upper()


def _generated_subjects() -> list[dict]:
    """Subjects that only exist because a standard clause needed one.

    The hand-written rule pack brings 62 subjects. A clause that is turned into
    a rule may need a subject nobody wrote down, and the model names it. Those
    subjects join the catalogue with the recognition cue the model gave, so the
    concurrent branches and the rule lookup can both see them.
    """
    if not GENERATED_RULES_PATH.exists():
        return []
    import json
    rules = json.loads(GENERATED_RULES_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, dict] = {}
    for rule in rules:
        if rule.get("subject_origin") != "new":
            continue
        name = str(rule.get("subject") or "").strip()
        if not name:
            continue
        entry = grouped.setdefault(name, {
            "trigger_id": f"GEN-{stable_id(name)}",
            "subject": name,
            "trigger_name": name,
            "cues": [],
            "visibility_tag": "外观",
            "recognition_mode": "vlm_visual",
            "observable_cues": [],
            "scenes": [],
            "families": [],
            "dimensions": [],
        })
        cue = str(rule.get("subject_cue") or "").strip()
        if cue and cue not in entry["observable_cues"]:
            entry["observable_cues"].append(cue)
            entry["cues"].append(cue)
        scene = str(rule.get("scene_hint") or "GENERAL_SITE")
        if scene not in entry["scenes"]:
            entry["scenes"].append(scene)
        family = SCENE_HINT_FAMILY.get(scene, "GENERAL")
        if family not in entry["families"]:
            entry["families"].append(family)
        dimension = str(rule.get("judge_dimension") or "").strip()
        if dimension and not any(d["dimension"] == dimension for d in entry["dimensions"]):
            entry["dimensions"].append({
                "dimension": dimension,
                "scenes": [scene],
                "boundary": rule.get("judge_boundary") or {},
            })
    out = []
    for entry in grouped.values():
        entry["scenes"] = sorted(entry["scenes"])
        entry["families"] = sorted(entry["families"])
        out.append(entry)
    out.sort(key=lambda item: item["trigger_name"])
    return out


MAIN_SYSTEM_RULES_PATH = BASE / "data" / "main_system" / "rules.json"
# 主系统规则包的场景提示 -> 我们的场景键
MAIN_SCENE_KEY = {
    "施工用电": "TEMPORARY_ELECTRICITY", "临时用电": "TEMPORARY_ELECTRICITY",
    "高处作业": "WORK_AT_HEIGHT", "基坑": "FOUNDATION_PIT",
    "起重吊装": "LIFTING_OPERATIONS", "塔吊": "TOWER_CRANE",
    "施工升降机": "CONSTRUCTION_HOIST",
    "悬挑式脚手架": "SCAFFOLD_CANTILEVER", "扣件式脚手架": "SCAFFOLD_COUPLER_TYPE",
    "盘扣式脚手架": "SCAFFOLD_DISC_BUCKLE",
    "附着式升降脚手架": "SCAFFOLD_ATTACHED_LIFTING",
}
# 每个主体最多带多少条规则进提示词
MAX_RULES_PER_SUBJECT = int(os.environ.get("MAX_RULES_PER_SUBJECT", "4"))


def _main_system_rules() -> list[dict]:
    """Rules from the full checker's runtime rule packs, keyed by subject."""
    if not MAIN_SYSTEM_RULES_PATH.exists():
        return []
    import json
    payload = json.loads(MAIN_SYSTEM_RULES_PATH.read_text(encoding="utf-8"))
    by_subject: dict[str, list[dict]] = {}
    for rule in payload.get("rules") or []:
        dimension = str(rule.get("judge_dimension") or "").strip()
        if not dimension:
            continue
        for trigger in rule.get("triggers") or []:
            name = str(trigger.get("trigger_name") or "").strip()
            if not name:
                continue
            by_subject.setdefault(name, []).append({
                "dimension": dimension,
                "boundary": rule.get("judge_boundary") or {},
                "scenes": [rule.get("scene_hint") or ""],
                "source": "main_system",
                "standard_code": rule.get("spec_code") or "",
                "article": str(rule.get("spec_clause") or ""),
            })
    return by_subject


def merge_main_system_into_subjects(subjects: list[dict]) -> list[dict]:
    """Fold the full checker's rules into the subject catalogue.

    The two catalogues share every subject we had, so the merge is by name.
    A subject that only exists in the full checker is added, with the scene
    family taken from the rule pack it came from.
    """
    extra = _main_system_rules()
    if not extra:
        return subjects
    index = {group["subject"]: group for group in subjects}
    for name, rules in extra.items():
        group = index.get(name)
        if group is None:
            scene_key = MAIN_SCENE_KEY.get(str(rules[0]["scenes"][0]), "GENERAL_SITE")
            group = {
                "trigger_id": f"MS-{stable_id(name)}",
                "subject": name,
                "trigger_name": name,
                "cues": [],
                "observable_cues": [],
                "dimensions": [],
                "scenes": [scene_key],
                "families": list(SCENE_FAMILY_BY_SCENE.get(scene_key, ["GENERAL"])),
                "origin": "main_system",
            }
            index[name] = group
            subjects.append(group)
        for rule in rules:
            if any(d["dimension"] == rule["dimension"] for d in group["dimensions"]):
                continue
            group["dimensions"].append(rule)
        # 主系统的主体是按具体场景分的，合并进来的规则也要挂到对应场景上
        for scene_key in {MAIN_SCENE_KEY.get(str(r["scenes"][0]), "GENERAL_SITE") for r in rules}:
            if scene_key not in group.get("scenes", []):
                group.setdefault("scenes", []).append(scene_key)
            for family in SCENE_FAMILY_BY_SCENE.get(scene_key, ["GENERAL"]):
                if family not in group.setdefault("families", []):
                    group["families"].append(family)
    # 每个主体只保留有限条"主系统补充规则"进提示词。
    # 规则包里人工整理的条目一律保留：那是质量最高的一批，不能被机器生成的挤掉。
    for group in subjects:
        own = [d for d in group["dimensions"] if d.get("source") != "main_system"]
        extra = [d for d in group["dimensions"] if d.get("source") == "main_system"]
        group["dimensions"] = own + extra[:MAX_RULES_PER_SUBJECT]
    return subjects


def render_subject_lines(subjects: list[dict], with_boundary: bool = True) -> list[str]:
    lines: list[str] = []
    for group in subjects:
        dims = [d["dimension"] for d in group["dimensions"]]
        label = "、".join(dims)
        if with_boundary:
            non_compliant = []
            for item in group["dimensions"]:
                text = shorten(str((item["boundary"] or {}).get("non_compliant") or ""), 30)
                if text and text not in non_compliant:
                    non_compliant.append(text)
            boundary = "；".join(non_compliant)
            if len(boundary) > 150:
                boundary = boundary[:147] + "…"
            lines.append(f"- 主体[{group['subject']}] {label}｜违规:{boundary}")
        else:
            lines.append(f"- 主体[{group['subject']}] {label}")
    return lines


def main() -> int:
    checklists = build()
    families = build_families(checklists)
    subjects = build_subjects()
    subjects = merge_main_system_into_subjects(subjects)
    subject_lines = render_subject_lines(subjects)
    lines = [
        '"""Per-scene inspection checklists generated from the rule pack.',
        "",
        "Generated by tools/build_scene_checklist.py. Each entry lists the unique",
        "check dimensions of one scene with the three-state judgement boundary of",
        "the rule, so the model works through the checklist instead of guessing",
        "which defects to look for.",
        "",
        "Only scenes in ENABLED_SCENES are injected into the prompt. Dimensions",
        "already governed by the object-and-state protocol point at that protocol",
        "instead of repeating a looser boundary.",
        '"""',
        "",
        "from typing import Any",
        "",
        "AUTO_CORE_DIMENSIONS = " + repr(AUTO_CORE_DIMENSIONS),
        "",
        "FAMILY_CORE_DIMENSIONS = " + repr(FAMILY_CORE_DIMENSIONS),
        "",
        "ENABLED_SCENES = " + repr(ENABLED_SCENES),
        "",
        "SCENE_FAMILIES = " + repr(SCENE_FAMILIES),
        "",
        "SCENE_SHORT_LABELS = " + repr(SCENE_SHORT_LABELS),
        "",
        "GLOBAL_DIMENSIONS = " + repr(tuple(GLOBAL_DIMENSIONS)),
        "",
        "SUBJECT_ANCHOR_ENABLED = " + repr(SUBJECT_ANCHOR_ENABLED),
        "",
        "SCENE_CHECKLISTS: dict[str, dict[str, Any]] = {",
    ]
    for scene, data in checklists.items():
        lines.append(f"    {scene!r}: {{")
        lines.append(f"        'rule_count': {data['rule_count']},")
        lines.append(f"        'dimension_count': {data['dimension_count']},")
        lines.append(f"        'enabled': {data['enabled']!r},")
        lines.append("        'lines': [")
        for item in data["lines"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("        'lines_with_anchor': [")
        for item in data["lines_with_anchor"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("SCENE_FAMILY_CHECKLISTS: dict[str, dict[str, Any]] = {")
    for family, data in families.items():
        lines.append(f"    {family!r}: {{")
        lines.append(f"        'label': {data['label']!r},")
        lines.append(f"        'scenes': {data['scenes']!r},")
        lines.append(f"        'scene_labels': {data['scene_labels']!r},")
        lines.append(f"        'rule_count': {data['rule_count']},")
        lines.append(f"        'dimension_count': {data['dimension_count']},")
        lines.append("        'lines': [")
        for item in data["lines"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("        'lines_with_anchor': [")
        for item in data["lines_with_anchor"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("        'core_lines': [")
        for item in data["core_lines"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("        'core_lines_with_anchor': [")
        for item in data["core_lines_with_anchor"]:
            lines.append(f"            {item!r},")
        lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    # 每个场景的主体清单行：场景已知时直接用它，等于把"该场景有哪些可查主体、
    # 每个主体查什么"一次给全，不必再让模型先分类。
    lines.append("SCENE_SUBJECT_LINES: dict[str, list[str]] = {")
    line_by_subject = {}
    for line in subject_lines:
        if line.startswith("- 主体["):
            name = line.split("]", 1)[0].split("[", 1)[1]
            line_by_subject[name] = line
    for scene in sorted(checklists):
        names = sorted({group["subject"] for group in subjects
                        if scene in group.get("scenes", [])})
        lines.append(f"    {scene!r}: {[line_by_subject[n] for n in names if n in line_by_subject]!r},")
    lines.append("}")
    lines.append("")
    lines.append("SCENE_SUBJECTS: dict[str, list[str]] = {")
    # 主体按它自己的场景归属来分，而不是按设备族：
    # 主系统的主体是按具体场景分的（悬挑式 32 个、扣件式 33 个），
    # 按族取会把四个场景的主体重叠在一起。
    for scene, data in checklists.items():
        names = sorted({group["subject"] for group in subjects
                        if scene in group.get("scenes", [])})
        lines.append(f"    {scene!r}: {names!r},")
    lines.append("}")
    lines.append("")
    lines.append("SUBJECT_CHECKLIST: list[dict[str, Any]] = [")
    for group in subjects:
        lines.append("    {")
        lines.append(f"        'trigger_id': {group['trigger_id']!r},")
        lines.append(f"        'subject': {group['subject']!r},")
        lines.append(f"        'cues': {group['cues']!r},")
        lines.append(f"        'scenes': {group['scenes']!r},")
        lines.append(f"        'families': {group['families']!r},")
        lines.append("        'dimensions': [")
        for item in group["dimensions"]:
            lines.append(f"            {{'dimension': {item['dimension']!r}, 'scenes': {item['scenes']!r}}},")
        lines.append("        ],")
        lines.append("    },")
    lines.append("]")
    lines.append("")
    lines.append("SUBJECT_CHECKLIST_LINES: list[str] = [")
    for item in subject_lines:
        lines.append(f"    {item!r},")
    lines.append("]")
    lines.append("")
    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {TARGET}")
    for scene, data in checklists.items():
        mark = "启用" if data["enabled"] else "未启用"
        size = sum(len(x) for x in data["lines"])
        print(f"  {scene:<28} {data['rule_count']:>3}条 -> {data['dimension_count']:>3}维度  {size:>5}字符  {mark}")
    print(f"  主体清单: {len(subjects)} 个主体 / {sum(len(x['dimensions']) for x in subjects)} 条规则 / "
          f"{sum(len(x) for x in subject_lines)} 字符")
    print("  -- 家族合并清单 --")
    for family, data in families.items():
        size = sum(len(x) for x in data["lines"])
        print(f"  {family:<28} {data['rule_count']:>3}条 -> {data['dimension_count']:>3}维度  {size:>5}字符  ({data['label']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
