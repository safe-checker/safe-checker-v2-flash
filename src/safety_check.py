#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek Flash construction safety quick checker.

DeepSeek Flash makes the visual safety decisions in one model call. Python
preprocesses the image, validates the structured response, resolves verified
regulation keys, converts bounding boxes, and formats the standard report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps, ImageStat

from scene_rules import (
    get_scene_profile,
    normalize_scene,
)
from verified_regulations import (
    VERIFIED_REGULATIONS,
    format_verified_regulation,
    verify_regulation,
)
try:  # 生成规则是可选扩展，索引文件缺失时照常运行
    from catalogs.generated_rules import GENERATED_BY_SCENE, GENERATED_BY_SUBJECT
except ImportError:  # pragma: no cover
    GENERATED_BY_SCENE, GENERATED_BY_SUBJECT = {}, {}

from catalogs.scene_checklists import (
    AUTO_CORE_DIMENSIONS,
    FAMILY_CORE_DIMENSIONS,
    SCENE_CHECKLISTS,
    SCENE_FAMILY_CHECKLISTS,
    SCENE_SUBJECT_LINES,
    SUBJECT_CHECKLIST,
    SUBJECT_CHECKLIST_LINES,
)


DEFAULT_PROMPT = (
    "请根据图片中清晰可见的施工现场情况，完成一次全面但快速的施工安全检查。"
    "覆盖人员防护、作业行为、空间关系、临边洞口、临时用电、机械设备、吊装、脚手架、"
    "基坑、消防、材料堆放和通道等可能存在的风险；由你根据图片和施工安全知识判断风险，"
    "说明直接证据，并依据施工安全规范知识匹配相关标准和具体条款。"
    "无法确认的不要臆测，写为“疑似违规”或“不可判断”。"
)


def _semantic_prompt_body(scene: str | None) -> str:
    scene_key = normalize_scene(scene)
    profile = get_scene_profile(scene_key)
    current_date = time.strftime("%Y年%m月")
    checklist_section = build_scene_checklist_section(scene_key)
    if not checklist_section:
        checklist_section = ""
    return f"""你是施工现场安全视觉检测模型。根据输入图片独立完成一次全面、快速的安全检查。
检查场景提示：{profile['label']}。该提示只是待验证线索，不能覆盖图片证据。先根据主体设备、结构、铭牌和用途标识识别实际场景；施工升降机/施工电梯的吊笼、层门、导轨、控制面板及“自动开门、禁止倚靠”等标识不得仅因出现电控柜而归为临时用电。用户未指定场景时按综合施工现场检查。

检查要求：先建立精简的 objects 对象清单，只收录安全判断所需的主体设备、关键部件、作业区域、危险源、防护设施以及被风险引用的对象，最多8个；不要收录与安全判断无关的背景物。再检查个人防护、作业行为、空间关系、临边洞口、高处、临时用电、机械、吊装、脚手架、基坑、消防、材料、通道及隔离警示等风险。只依据图片直接可见证据；不得把阴影、反光、模糊、遮挡或画面裁切推断成违规。未入镜不等于缺失，空间关系须有清楚几何证据。对开/闭、干/湿、有/无、内/外、存在/缺失、占用/相邻等状态，必须确认目标身份、部件归属和边界，给出至少2条相互独立的直接视觉线索，并排除一个最合理的相反解释；否则只能 SUSPECTED 或 NOT_ASSESSABLE。静态图片不能证明资质、票证、检测记录、承载力、功能试验或精确距离。

若用户同时提供全图和局部图，第一张始终是整图，后续局部图只用于补充细节；必须先用整图建立空间关系，再用局部图确认部件状态，不能把局部裁剪边界当作原图边界，也不能因局部图看不全而推断整图中目标缺失。

对象与状态协议：
1. 配电箱主箱门、内层面板、检修门、插座盖和插口是不同部件。主箱门打开必须同时看见主门门板发生角度位移以及主箱开口/内部空间；箱体侧面、门缝、圆形插口或打开的插座盖不能作为主箱门打开证据。箱门关闭但未上锁，应判“未锁闭”，不得判“箱门未关闭”。
2. 积水必须看见连续液体表面，并至少具有水面边界、镜面反射、波纹或对周围物体的倒影/遮挡关系中的两项。单一深色区域、土壤色差、阴影或局部反光不是积水。
3. 判断护栏/围栏缺失前，先逐项盘点上横杆、中横杆、立杆、挡脚板或网片；已有横杆不得因角度、遮挡或与背景同色而写成“未见护栏”。只允许指出实际缺失的具体构件。
4. 箱门内侧固定的系统图、接线图、责任牌、标签和说明页属于设备文件，不是箱内杂物。只有独立放置、与设备无固定关系且明确无关的物品才可判为杂物。
5. 空间占用必须分别定位设备操作区和障碍物，证明二者边界发生实际重叠或阻断通道；仅在设备附近、透视上重叠或位于不同深度，不等于占用。

人物优先检查协议：先逐人建立 persons 清单并分配 P1、P2 等编号，再判断人物风险。对每人按“头部→左手→右手→躯干→腰部→左脚→右脚”的顺序逐部位观察。装备必须按正确佩戴部位检查：安全帽对应头部，左/右手套分别对应左/右手，反光背心对应躯干，安全带对应躯干和腰部；颜色相近、手持物、背景材料或局部遮挡不得当成佩戴或未佩戴证据。左手套和右手套必须分别记录；相关部位不完整清晰时不得声称“未佩戴”，被钢筋、工具、身体或画面边缘遮挡的一侧写 uncertain。手套、面罩、护目镜、反光背心等任务型防护用品只有在具体作业动作、独立危险对象和对应危害均清晰可见，且有明确适用条款时才可判缺失；仅因人员在工地、高处或车辆附近不得判定缺少这些装备。白天照片中背心颜色、反光条外观或未出现强反光不能证明反光性能不合格，也不能据此判定穿戴不规范。每条 item 只描述一个确定状态，禁止使用“未佩戴或佩戴不规范”等二选一表述。空间位置必须以身体锚点和参照物锚点判断，优先检查双脚接触面、支撑面、临边/洞口边界、可见高差和前后遮挡关系。抬腿、跨越材料、站在低矮材料堆旁不等于攀爬或高处作业；只有人员明确位于存在可见坠落高差的高处作业面，才能写 ELEVATED_WITH_FALL_RISK。靠近钢丝绳、车辆或静置材料不等于处于吊物下方；只有吊物与支撑面明显分离、确实处于悬空或移动状态，且人员锚点位于吊物垂直投影或回转路径内，才能明确判定吊装危险区人员风险。

每个风险输出一个紧贴可见违规证据的范围框。坐标基于整张输入图片，采用0到1000归一化整数，原点在左上角，格式为 [x1,y1,x2,y2]，其中x向右、y向下，x1/y1为左上角，x2/y2为右下角。范围框应包住实际违规对象/证据，不要框整张图或无关区域。若一条风险涉及彼此分离的多个位置，拆成多个风险项分别定位。无法从图中可靠定位时 bbox_2d_1000 必须为 null，禁止猜坐标；有可定位证据时必须给有效坐标。

{checklist_section}
法规要求：当前日期为{current_date}。对每个风险，先依据施工安全规范知识给出最可能的中国现行标准候选，优先使用当前仍有效的最新版本，不要在已知存在新版时引用被替代的旧版。输出规范全称、标准编号、具体条款号和条文要点；不能确认标准真实性、条号或适用条件时将 match_status 写 NEEDS_VERIFICATION，禁止编造法规。法规匹配范围不得局限于临时用电，应覆盖图片实际涉及的高处、吊装、脚手架、基坑、机械、消防和个人防护等场景。程序随后只负责将候选与本地已核验现行条款目录比对；只有目录核验成功的条款才能作为正式引用，模型候选不能单独支撑明确违规；模型仍负责识图和安全判断。

risk_category 使用下列最接近的场景键：GENERAL_SITE、TEMPORARY_ELECTRICITY、WORK_AT_HEIGHT、FOUNDATION_PIT、LIFTING_OPERATIONS、TOWER_CRANE、CONSTRUCTION_HOIST、SCAFFOLD_COUPLER_TYPE、SCAFFOLD_DISC_BUCKLE、SCAFFOLD_CANTILEVER、SCAFFOLD_ATTACHED_LIFTING。risk_key 使用稳定、简短、大写英文下划线编码，表达根本风险，不包含位置和后果，例如 PERSON_UNDER_SUSPENDED_LOAD、CABLE_ON_GROUND；没有合适固定名称时自行给出稳定编码。

状态：CLEAR表示违规状态有直接清晰证据，item和evidence中不得出现“疑似、可能、无法确认、无法判断”等不确定措辞；SUSPECTED表示存在可见线索但仍有不确定性；NOT_ASSESSABLE表示图片不能判断，此时item应描述“某状态无法判断”，不要把未确认事实写成确定违规名称。
"""

_OUTPUT_TAIL_FULL = """只输出一个合法JSON对象，不输出Markdown、代码围栏或解释。按严重度最多输出{MAX_RISK_ITEMS}条最重要风险；所有中文描述保持简短，单条 evidence 不超过60字，每个线索数组最多2项，不重复解释。格式：
{{
  "scene_type": "根据图片识别的场景键",
  "scene": "场景中文名称",
  "scene_confidence": 0.0,
  "scene_evidence": ["主体设备线索1", "主体设备线索2"],
  "objects": [
    {{
      "object_id": "O1",
      "object_type": "主体设备或部件类型",
      "parent_id": "父对象ID或null",
      "bbox_2d_1000": [100, 100, 300, 800],
      "visibility": "complete/partial/occluded/not_visible",
      "identity_cues": ["身份线索1", "身份线索2"],
      "observed_state": "直接看见的状态，不作风险推断"
    }}
  ],
  "persons": [
    {{
      "person_id": "P1",
      "bbox_2d_1000": [100, 100, 300, 800],
      "body_visibility": {{"head": "complete/partial/occluded/not_visible", "left_hand": "...", "right_hand": "...", "torso": "...", "waist": "...", "left_foot": "...", "right_foot": "..."}},
      "hand_ppe_states": {{"left_glove": "worn/not_worn/improper/uncertain/not_applicable", "right_glove": "..."}},
      "ppe_states": {{"helmet": "worn/not_worn/improper/uncertain/not_applicable", "gloves": "...", "reflective_vest": "...", "safety_harness": "..."}},
      "action": "人物正在进行的可见动作",
      "support_surface": "双脚实际接触或支撑的物体/地面；看不清写unknown",
      "elevation_state": "GROUND_LEVEL/LOW_OBSTACLE/ELEVATED_WITH_FALL_RISK/UNCERTAIN",
      "spatial_cues": ["双脚与支撑面的直接关系", "人物与参照物的直接关系"]
    }}
  ],
  "issues": [
    {{
      "status": "CLEAR/SUSPECTED/NOT_ASSESSABLE",
      "item": "简明违规项",
      "risk_category": "场景键",
      "risk_key": "稳定的大写英文风险键",
      "assessment_type": "PPE/SPATIAL/OBJECT_STATE/MATERIAL_STATE/PRESENCE_ABSENCE/OTHER",
      "claim_type": "OPEN_CLOSED/PRESENCE_ABSENCE/MATERIAL_SURFACE/SPATIAL_OCCUPANCY/OBJECT_IDENTITY/FOREIGN_OBJECT/PPE/OTHER",
      "object_ids": ["O1"],
      "person_ids": ["P1"],
      "ppe_item": "helmet/gloves/reflective_vest/safety_harness/other/not_applicable",
      "focus_body_parts": ["head/left_hand/right_hand/torso/waist/left_foot/right_foot"],
      "spatial_relation": {{"subject_anchor": "身体锚点", "reference_object": "参照物", "reference_anchor": "参照物锚点", "both_anchors_visible": true, "elevation_state": "GROUND_LEVEL/LOW_OBSTACLE/ELEVATED_WITH_FALL_RISK/UNCERTAIN", "reference_state": "STATIC/SUSPENDED/MOVING/UNCERTAIN", "person_zone": "UNDER_LOAD/SWING_PATH/ADJACENT/OUTSIDE/UNCERTAIN", "cues": ["空间线索1", "空间线索2"]}},
      "claim_validation": {{"target_identity_confirmed": true, "component_boundary_complete": true, "inspection_region_complete": true, "positive_cues": ["正向证据1", "正向证据2"], "alternative_explanation": "最合理相反解释", "alternative_excluded": true, "subject_anchor": "主体锚点", "reference_anchor": "参照锚点", "relation": "OVERLAP/BLOCKING/CONTACT/ADJACENT/SEPARATE/UNCERTAIN"}},
      "target": "违规对象",
      "position": "图中方位",
      "evidence": "图片中直接可见的现象",
      "visual_cues": ["直接视觉线索1", "直接视觉线索2"],
      "regulation": {{
        "standard_name": "规范或标准全称",
        "standard_code": "标准编号",
        "article": "具体条款号",
        "clause_summary": "与该风险直接相关的条文要点",
        "match_reason": "该条款与图片风险的对应关系",
        "match_status": "MATCHED/NEEDS_VERIFICATION"
      }},
      "confidence": 0.0,
      "target_visibility": "complete/partial/occluded/not_visible",
      "evidence_level": "direct/partial/occluded/not_visible/ambiguous",
      "citation_confidence": "high/medium/low/unknown",
      "bbox_2d_1000": [100, 200, 400, 650]
    }}
  ]
}}
无风险时 issues 为空数组。"""


_OUTPUT_TAIL_COMPACT = """只输出一个合法JSON对象，不输出Markdown、代码围栏或解释。按严重度最多输出{MAX_RISK_ITEMS}条最重要风险。
使用下面的短键输出，不要展开成完整字段名；中文一律短句：evidence 不超过40字，item 不超过25字，target 不超过12字，position 不超过8字，alt 不超过15字，sp/cv 里的锚点不超过8字，cu/vc 每项不超过10字且最多2项。objs 最多6个，ppl 只列与风险或防护判断相关的人。
枚举值必须写全称，例如 status 写 CLEAR / SUSPECTED / NOT_ASSESSABLE，vis 写 complete / partial / occluded / not_visible，pp/hp 写 worn / not_worn / improper / uncertain / not_applicable，el 写 GROUND_LEVEL / LOW_OBSTACLE / ELEVATED_WITH_FALL_RISK / UNCERTAIN，rs 写 STATIC / SUSPENDED / MOVING / UNCERTAIN，pz 写 UNDER_LOAD / SWING_PATH / ADJACENT / OUTSIDE / UNCERTAIN，rel 写 OVERLAP / BLOCKING / CONTACT / ADJACENT / SEPARATE / UNCERTAIN，ev 写 direct / partial / occluded / not_visible / ambiguous，cc 写 high / medium / low / unknown。
reg 只写标准编号与条号两项，不要写标准全称和条文要点，程序用本地已核验目录回填；无法确定条号时写 "a": "NEEDS_VERIFICATION"。
格式：
{{
  "st": "场景键", "sc": "场景中文名", "sconf": 0.0, "sev": ["线索1", "线索2"],
  "objs": [
    {{"id": "O1", "t": "主体或部件类型", "b": [100, 100, 300, 800], "vis": "complete", "s": "≤20字可见状态"}}
  ],
  "ppl": [
    {{"id": "P1", "b": [100, 100, 300, 800],
     "bv": {{"head": "complete", "lh": "partial", "rh": "partial", "tr": "complete", "wa": "partial", "lf": "complete", "rf": "complete"}},
     "hp": {{"lg": "not_applicable", "rg": "not_applicable"}},
     "pp": {{"helmet": "worn", "gloves": "not_applicable", "vest": "not_applicable", "harness": "not_applicable"}},
     "ac": "≤15字动作", "sup": "≤15字支撑面", "el": "GROUND_LEVEL"}}
  ],
  "iss": [
    {{
      "s": "CLEAR", "i": "≤25字违规项", "cat": "场景键", "key": "UPPER_SNAKE_KEY",
      "at": "PPE", "ct": "PPE", "oid": ["O1"], "pid": ["P1"], "ppe": "helmet", "bp": ["head"],
      "sp": {{"sa": "≤8字主体锚点", "ro": "≤8字参照物", "bv": true, "el": "GROUND_LEVEL", "rs": "STATIC", "pz": "OUTSIDE", "cu": ["≤10字", "≤10字"]}},
      "cv": {{"id": true, "bd": true, "rg": true, "cu": ["≤10字", "≤10字"], "alt": "≤15字相反解释", "ax": true, "sa": "≤8字", "ra": "≤8字", "rel": "ADJACENT"}},
      "t": "≤12字对象", "pos": "≤8字方位", "e": "≤40字可见证据", "vc": ["≤10字", "≤10字"],
      "reg": {{"c": "标准编号", "a": "条号"}},
      "cf": 0.0, "tv": "complete", "ev": "direct", "cc": "medium", "b": [100, 200, 400, 650]
    }}
  ]
}}
无风险时 iss 为空数组。"""


SCENE_LABELS_ZH = {
    "SCAFFOLD_CANTILEVER": "悬挑式脚手架",
    "SCAFFOLD_COUPLER_TYPE": "扣件式钢管脚手架",
    "SCAFFOLD_DISC_BUCKLE": "承插型盘扣式钢管脚手架",
    "SCAFFOLD_ATTACHED_LIFTING": "附着式升降脚手架",
    "FOUNDATION_PIT": "基坑工程",
    "TOWER_CRANE": "塔式起重机",
    "CONSTRUCTION_HOIST": "施工升降机",
    "LIFTING_OPERATIONS": "起重吊装",
    "TEMPORARY_ELECTRICITY": "临时用电/配电箱",
    "WORK_AT_HEIGHT": "高处作业",
    "GENERAL_SITE": "施工现场综合",
    "SCAFFOLD_FAMILY": "脚手架（悬挑式/扣件式/盘扣式/附着升降式）",
    "LIFTING_FAMILY": "起重与升降设备（塔式起重机/施工升降机/起重吊装）",
}
CHECKLIST_USAGE = (
    "清单是检查范围的提示，不是必须输出的条目：逐项核对后，只输出证据充分、确实成立的风险；"
    "未发现问题的项目不要写进结果，也不要为凑满清单而报告不成立的风险。"
    "标注“判定：按…”的行以对应协议原文为准，协议严格于清单；"
    "清单与前面的对象与状态协议、人物优先检查协议、状态定义同时生效。"
)


_ANCHOR_RE = re.compile(r"(?m)^(- )\[[^\]]{1,20}\] ")


# 主体分组：把主体按设备族切开，同一张图可以由多路并发各查一组。
# 这是主系统"逐条核查"在 8 秒预算下的压缩形态——L4 的 N 次串行核查做不了，
# 但"按主体切开、并发查"的时间上限等于最慢的一路，不是 N 路之和。
SUBJECT_GROUP_FAMILIES = [
    ("脚手架", ("SCAFFOLD",)),
    ("起重与升降设备", ("LIFTING",)),
    ("临时用电", ("ELECTRICITY",)),
    ("基坑·高处·综合", ("PIT", "HEIGHT", "GENERAL")),
]
# 人员防护主体归入最后一组，避免每一路都重复报同一件事
PPE_SUBJECT = "图片中人员"


def _group_index() -> tuple[int | None, int]:
    """Read the concurrent branch index from the environment."""
    raw = os.environ.get("CHECKLIST_SUBJECT_GROUP", "").strip()
    if not raw:
        return None, 1
    try:
        index_text, total_text = raw.split("/", 1)
        index, total = int(index_text), int(total_text)
    except ValueError:
        return None, 1
    if total <= 1 or index < 0 or index >= total:
        return None, 1
    return index, total


def _render_subject_section(subject_lines: list[str], header: str = "") -> str:
    lines = [
        header or ("本场景主体清单：先在图中逐条确认下列主体是否存在。只核对确实存在的主体的条目，"
                   "主体不存在直接跳过，不得据推测判定未入镜的主体缺失。"),
        "每条格式为「主体[名称] 检查内容｜违规:违规情形」。",
    ]
    lines.extend(subject_lines)
    lines.append(CHECKLIST_USAGE)
    return "\n".join(lines)


AUTO_SUBJECT_BUDGET = 12000


def _auto_subject_lines() -> list[str]:
    """The subject lines an AUTO run may use, chosen inside a character budget.

    The catalogue now holds 355 subjects. Handing every branch all of them would
    push the prompt past what eight seconds allows, so the subjects are picked in
    priority order until the budget is spent: the hand-written rule pack first,
    then the generated rules, then the full checker's rules.
    """
    budget = env_int("AUTO_SUBJECT_BUDGET", AUTO_SUBJECT_BUDGET)
    order = []
    for prefix in ("", "GEN-", "MS-"):
        for group in SUBJECT_CHECKLIST:
            if str(group["trigger_id"]).startswith(prefix) and group not in order:
                order.append(group)
    lines = []
    used = 0
    for group in order:
        row = next((line for line in SUBJECT_CHECKLIST_LINES
                    if line.startswith(f"- 主体[{group['subject']}]")), None)
        if not row:
            continue
        if used + len(row) > budget:
            continue
        lines.append(row)
        used += len(row)
    return lines


def _subject_group_lines() -> list[str] | None:
    """Return the subject lines of one concurrent branch, or None for all."""
    raw = os.environ.get("CHECKLIST_SUBJECT_GROUP", "").strip()
    if not raw:
        return None
    try:
        index_text, total_text = raw.split("/", 1)
        index = int(index_text)
        total = int(total_text)
    except ValueError:
        return None
    if total <= 1 or index < 0 or index >= total:
        return None
    if total > len(SUBJECT_GROUP_FAMILIES):
        return None
    groups = SUBJECT_GROUP_FAMILIES[:total]
    if total < len(SUBJECT_GROUP_FAMILIES):
        # 合并尾部，保证每个主体都有归属
        head = groups[:-1]
        tail = groups[-1]
        families = set()
        for _, fams in groups[total - 1:]:
            families |= set(fams)
        groups = head + [(tail[0], tuple(sorted(families)))]
    wanted = set(groups[index][1])
    lines = []
    for group in SUBJECT_CHECKLIST:
        subject = group["subject"]
        families = set(group["families"])
        if subject == PPE_SUBJECT:
            if index != total - 1:
                continue
        elif not (families & wanted):
            continue
        for line in SUBJECT_CHECKLIST_LINES:
            if line.startswith(f"- 主体[{subject}]"):
                lines.append(line)
                break
    return lines


def _generated_rule_lines(rules: list[dict], limit: int, max_chars: int) -> list[str]:
    """Render generated rules for the prompt.

    Each line keeps the same shape as the hand-curated checklist so the model
    reads one format: dimension, then what counts as a violation.
    """
    lines: list[str] = []
    used = 0
    for rule in rules[: max(0, limit)]:
        boundary = rule.get("judge_boundary") or {}
        line = (f"- [{rule.get('standard_code', '')} {rule.get('article', '')}] "
                f"{rule.get('judge_dimension', '')}"
                f"｜违规:{str(boundary.get('non_compliant', ''))[:34]}")
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return lines


def _generated_for_scene(scene_key: str) -> list[str]:
    """Generated rules for one scene, if the opt-in switch is on."""
    if os.environ.get("USE_GENERATED_RULES", "0").strip() != "1":
        return []
    limit = env_int("GENERATED_RULES_LIMIT", 12)
    max_chars = env_int("GENERATED_RULES_MAX_CHARS", 1400)
    rules = GENERATED_BY_SCENE.get(scene_key, [])
    if not rules and scene_key.startswith("SCAFFOLD"):
        rules = GENERATED_BY_SCENE.get("SCAFFOLD_FAMILY", [])
    if not rules and scene_key.startswith("LIFTING"):
        rules = GENERATED_BY_SCENE.get("LIFTING_FAMILY", [])
    return _generated_rule_lines(rules, limit, max_chars)


def _generated_for_subjects(subjects: list[str]) -> list[str]:
    """Generated rules bound to the subjects a branch is responsible for."""
    if os.environ.get("USE_GENERATED_RULES", "0").strip() != "1":
        return []
    limit = env_int("GENERATED_RULES_LIMIT", 12)
    max_chars = env_int("GENERATED_RULES_MAX_CHARS", 1400)
    rules: list[dict] = []
    for subject in subjects:
        rules.extend(GENERATED_BY_SUBJECT.get(subject, []))
    return _generated_rule_lines(rules, limit, max_chars)


def build_scene_checklist_section(scene_key: str) -> str:
    """Render the inspection checklist for the scene, if one is available.

    A confirmed scene gets its full checklist. When the scene is not known yet
    the section lists the observable subjects, because the whole checklist of
    every scene would not fit in the prompt.
    """
    return _build_scene_checklist_section(scene_key)


def _build_scene_checklist_section(scene_key: str) -> str:
    # 主体锚点（[主体名] 前缀）默认关闭：94 张配对实测开 81.7%、关 84.0%，方向为负。
    # 设 CHECKLIST_SUBJECT_ANCHOR=1 可以再打开做对照。
    anchor_key = "lines_with_anchor" if os.environ.get("CHECKLIST_SUBJECT_ANCHOR", "0").strip() == "1" else "lines"
    core_key = "core_lines_with_anchor" if anchor_key == "lines_with_anchor" else "core_lines"
    enabled = {key: data for key, data in SCENE_CHECKLISTS.items() if data.get("enabled")}
    # USE_MAIN_SYSTEM_RULES=0 回到合并前的清单，用于对照实验
    use_main = os.environ.get("USE_MAIN_SYSTEM_RULES", "1").strip() != "0"
    if scene_key and scene_key != "AUTO":
        # 场景已知：取该场景的主体清单。
        #
        # SCENE_CHECKLIST_SCOPE 决定这份清单在并发时怎么分：
        #   split（默认）—— 按路数均分，每路只查自己那 1/N。
        #   full        —— 每路都拿全量清单，本路负责的那部分排在前面。
        #
        # split 的问题在 140 张陌生集上暴露得很清楚：施工升降机 25 个主体
        # 切成 4 份后每路只剩 6-7 个，标准答案落在别的 3/4 路上就直接漏检。
        # 140 张实测（排除 API 故障样本）：split 命中 27.4%，full 见 SCOPE_FULL。
        # 并发分摊的是注意力，不是覆盖范围；覆盖必须留给每一路。
        rows = (SCENE_SUBJECT_LINES.get(scene_key) or []) if use_main else []
        index, total = _group_index()
        scope = os.environ.get("SCENE_CHECKLIST_SCOPE", "split").strip().lower() or "split"
        label = SCENE_LABELS_ZH.get(scene_key, scene_key)
        if rows and index is not None and scope == "full":
            picked = [row for i, row in enumerate(rows) if i % total == index]
            header = (f"本场景检查清单（{label}，共{len(rows)}个可查主体，全部列出；"
                      f"其中标「本路重点」的{len(picked)}个本路逐条核对，"
                      f"其余主体一并看过一遍，确有问题的也要报）：")
            lines = [header, "每条格式为「主体[名称] 检查内容｜违规:违规情形」。"]
            for row in picked:
                lines.append(row.replace("- 主体[", "- 【本路重点】主体[", 1))
            for row in rows:
                if row not in picked:
                    lines.append(row)
            lines.append(CHECKLIST_USAGE)
            return "\n".join(lines)
        if rows and index is not None:
            picked = [row for i, row in enumerate(rows) if i % total == index]
            return _render_subject_section(
                picked,
                header=f"本场景检查清单（{label}，共{len(rows)}个可查主体，本路负责{len(picked)}个，逐项核对）：")
    if not enabled:
        return ""
    if scene_key and scene_key in enabled:
        data = enabled[scene_key]
        label = SCENE_LABELS_ZH.get(scene_key, scene_key)
        subject_rows = SCENE_SUBJECT_LINES.get(scene_key) or [] if use_main else []
        if subject_rows:
            # 场景已知时，用"该场景的可查主体 + 每个主体查什么"组织，
            # 比只给场景维度的清单覆盖更广，也和 AUTO 路径用同一套主体。
            lines = [f"本场景检查清单（{label}，{len(subject_rows)}个可查主体，逐项核对）："]
            lines.extend(subject_rows)
        else:
            lines = [f"本场景检查清单（{label}，共{data['dimension_count']}项，逐项核对）："]
            lines.extend(data[anchor_key])
        extra = _generated_for_scene(scene_key)
        if extra:
            lines.append("补充检查项（来自标准条款，与上面同样适用）：")
            lines.extend(extra)
        lines.append(CHECKLIST_USAGE)
        return "\n".join(lines)
    if scene_key in SCENE_FAMILY_CHECKLISTS:
        # 调用方直接指定了家族：用家族合并清单，子类特有维度带适用条件
        data = SCENE_FAMILY_CHECKLISTS[scene_key]
        lines = [
            f"本场景检查清单（{data['label']}家族，含{'/'.join(data['scene_labels'])}，"
            f"共{data['dimension_count']}项，逐项核对）：",
        ]
        lines.extend(data[anchor_key])
        extra = _generated_for_scene(scene_key)
        if extra:
            lines.append("补充检查项（来自标准条款，与上面同样适用）：")
            lines.extend(extra)
        lines.append(CHECKLIST_USAGE)
        return "\n".join(lines)
    if scene_key and scene_key != "AUTO":
        # 场景已确定但该场景尚未开放清单：不注入，避免塞入别的场景的清单
        return ""
    # 场景未确定。三种给法可以用 CHECKLIST_AUTO_MODE 切换，用来做对照实验。
    # 默认用主体清单。94 张配对实测：主体清单 72.1% > 场景预览 67.7% > 家族清单 62.9%。
    mode = os.environ.get("CHECKLIST_AUTO_MODE", "subject").strip().lower() or "subject"
    if mode == "subject":
        # 主体清单：主系统 trigger→rule 机制在单次调用里的形态。
        # 不要求模型先命名场景，而是先在图中找出出现的主体，只核对出现主体对应的条目。
        # 主体路径：生成主体已经作为主体清单行进来了，不再单独注入，避免重复
        index, total = _group_index()
        if index is not None:
            picked = [row for i, row in enumerate(_auto_subject_lines()) if i % total == index]
            return _render_subject_section(
                picked,
                header="本场景主体清单（场景未定，先确认下列主体是否存在；本路负责其中一部分）：")
        return _render_subject_section(_auto_subject_lines())
    if mode == "scene":
        # 旧路径：逐场景给核心维度名称，不给判定边界。
        names = "、".join(SCENE_LABELS_ZH.get(key, key) for key in sorted(enabled))
        lines = [
            "本场景检查清单：图片场景尚未确定，先判断实际场景，再按该场景的清单逐项核对。",
            f"已开放清单的场景：{names}。",
        ]
        for key, data in enabled.items():
            label = SCENE_LABELS_ZH.get(key, key)
            preview = [line.split("｜")[0] for line in data[anchor_key][:AUTO_CORE_DIMENSIONS]]
            lines.append(f"【{label}】核心维度{len(preview)}项（确认属于该场景后按完整清单核对）：")
            lines.extend(preview)
        lines.append(CHECKLIST_USAGE)
        return "\n".join(lines)
    # 默认：按家族给核心维度。
    # 模型把子类判错的概率不低，但把家族判对的比例很高，
    # 所以 AUTO 路径以家族为单位给清单，子类特有维度标注适用条件。
    lines = [
        "本场景检查清单：图片场景尚未确定。先判断图片属于下列哪个家族，再按该家族的核心维度逐项核对。",
        "标注“仅XX适用”的维度在无法确定子类时跳过；只核对与图中实际存在主体相关的维度。",
    ]
    for key, data in SCENE_FAMILY_CHECKLISTS.items():
        core = data[core_key][:FAMILY_CORE_DIMENSIONS] or data[anchor_key][:FAMILY_CORE_DIMENSIONS]
        lines.append(f"【{data['label']}】（{'/'.join(data['scene_labels'])}）核心维度{len(core)}项：")
        lines.extend(core)
    lines.append(CHECKLIST_USAGE)
    return "\n".join(lines)


def max_risk_items() -> int:
    """How many risk items the model may report.

    Three was the original cap. The 94-image run shows the model fills all
    three slots with the most prominent defects and drops the subtler one the
    test set labelled, while 75% of the extra items turn out to be real. The
    cap is therefore a variable, not a constant.
    """
    value = env_int("MAX_RISK_ITEMS", 3)
    return max(1, min(value, 8))


def build_single_vision_system_prompt(scene: str | None, compact: bool = True) -> str:
    """Semantic rules stay identical in both modes; only the output template differs."""
    limit = max_risk_items()
    tail = (_OUTPUT_TAIL_COMPACT if compact else _OUTPUT_TAIL_FULL).replace("{MAX_RISK_ITEMS}", str(limit))
    return _semantic_prompt_body(scene) + tail



_COMPACT_TOP_KEYS = {
    "st": "scene_type", "sc": "scene", "sconf": "scene_confidence", "sev": "scene_evidence",
    "objs": "objects", "ppl": "persons", "iss": "issues",
}
_COMPACT_OBJECT_KEYS = {
    "id": "object_id", "t": "object_type", "b": "bbox_2d_1000",
    "vis": "visibility", "s": "observed_state",
}
_COMPACT_PERSON_KEYS = {
    "id": "person_id", "b": "bbox_2d_1000", "bv": "body_visibility", "hp": "hand_ppe_states",
    "pp": "ppe_states", "ac": "action", "sup": "support_surface", "el": "elevation_state",
}
_COMPACT_BODY_KEYS = {
    "head": "head", "lh": "left_hand", "rh": "right_hand", "tr": "torso",
    "wa": "waist", "lf": "left_foot", "rf": "right_foot",
}
_COMPACT_GLOVE_KEYS = {"lg": "left_glove", "rg": "right_glove"}
_COMPACT_PPE_KEYS = {"helmet": "helmet", "gloves": "gloves", "vest": "reflective_vest", "harness": "safety_harness"}
_COMPACT_ISSUE_KEYS = {
    "s": "status", "i": "item", "cat": "risk_category", "key": "risk_key",
    "at": "assessment_type", "ct": "claim_type", "oid": "object_ids", "pid": "person_ids",
    "ppe": "ppe_item", "bp": "focus_body_parts", "sp": "spatial_relation", "cv": "claim_validation",
    "t": "target", "pos": "position", "e": "evidence", "vc": "visual_cues", "reg": "regulation",
    "cf": "confidence", "tv": "target_visibility", "ev": "evidence_level", "cc": "citation_confidence",
    "b": "bbox_2d_1000",
}
_COMPACT_SPATIAL_KEYS = {
    "sa": "subject_anchor", "ro": "reference_object", "ra": "reference_anchor",
    "bv": "both_anchors_visible", "el": "elevation_state", "rs": "reference_state",
    "pz": "person_zone", "cu": "cues",
}
_COMPACT_VALIDATION_KEYS = {
    "id": "target_identity_confirmed", "bd": "component_boundary_complete",
    "rg": "inspection_region_complete", "cu": "positive_cues",
    "alt": "alternative_explanation", "ax": "alternative_excluded",
    "sa": "subject_anchor", "ra": "reference_anchor", "rel": "relation",
}
_COMPACT_REGULATION_KEYS = {
    "c": "standard_code", "a": "article", "n": "standard_name",
    "s": "clause_summary", "r": "match_reason", "st": "match_status",
}


def _compact_rename(value: Any, mapping: dict[str, str]) -> Any:
    """Rename short keys to canonical names, keeping already-canonical keys."""
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        canonical = mapping.get(key, key)
        if canonical in out:
            continue
        out[canonical] = item
    return out


def expand_compact_result(data: Any) -> dict[str, Any]:
    """Expand the short-key model output into the canonical internal structure.

    Long canonical keys are accepted as well, so a model that ignores the short
    key convention still produces a usable report.
    """
    if not isinstance(data, dict):
        return {}
    result = _compact_rename(data, _COMPACT_TOP_KEYS)
    objects = result.get("objects")
    if isinstance(objects, list):
        result["objects"] = [_compact_rename(o, _COMPACT_OBJECT_KEYS) for o in objects]
    persons = result.get("persons")
    if isinstance(persons, list):
        expanded = []
        for person in persons:
            person = _compact_rename(person, _COMPACT_PERSON_KEYS)
            if isinstance(person.get("body_visibility"), dict):
                person["body_visibility"] = _compact_rename(person["body_visibility"], _COMPACT_BODY_KEYS)
            if isinstance(person.get("hand_ppe_states"), dict):
                person["hand_ppe_states"] = _compact_rename(person["hand_ppe_states"], _COMPACT_GLOVE_KEYS)
            if isinstance(person.get("ppe_states"), dict):
                person["ppe_states"] = _compact_rename(person["ppe_states"], _COMPACT_PPE_KEYS)
            expanded.append(person)
        result["persons"] = expanded
    issues = result.get("issues")
    if isinstance(issues, list):
        expanded_issues = []
        for issue in issues:
            issue = _compact_rename(issue, _COMPACT_ISSUE_KEYS)
            if isinstance(issue.get("spatial_relation"), dict):
                issue["spatial_relation"] = _compact_rename(issue["spatial_relation"], _COMPACT_SPATIAL_KEYS)
            if isinstance(issue.get("claim_validation"), dict):
                issue["claim_validation"] = _compact_rename(issue["claim_validation"], _COMPACT_VALIDATION_KEYS)
            if isinstance(issue.get("regulation"), dict):
                issue["regulation"] = _compact_rename(issue["regulation"], _COMPACT_REGULATION_KEYS)
            expanded_issues.append(issue)
        result["issues"] = expanded_issues
    return result


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default



def analyze_image_quality(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return analyze_pil_image_quality(image)


def analyze_pil_image_quality(image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    sample = image.convert("RGB")
    sample.thumbnail((320, 320))
    gray = sample.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0])
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge)
    sharpness = float(edge_stat.var[0])
    # Pillow 10 起 getdata 逐步废弃，新版改用 get_flattened_data，两种都兼容
    pixels = gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else list(gray.getdata())
    total = max(1, len(pixels))
    overexposed_ratio = sum(1 for p in pixels if p >= 245) / total
    underexposed_ratio = sum(1 for p in pixels if p <= 15) / total

    clarity = "clear"
    if sharpness < 140:
        clarity = "blurry"
    elif sharpness < 260:
        clarity = "uncertain"

    glare = overexposed_ratio > 0.04 or brightness > 210
    dark = underexposed_ratio > 0.25 or brightness < 40
    notes: list[str] = []
    if clarity != "clear":
        notes.append("清晰度不足")
    if glare:
        notes.append("存在强光或过曝")
    if dark:
        notes.append("存在欠曝区域")

    return {
        "width": width,
        "height": height,
        "clarity": clarity,
        "brightness": round(brightness, 1),
        "sharpness": round(sharpness, 1),
        "overexposed_ratio": round(overexposed_ratio, 4),
        "underexposed_ratio": round(underexposed_ratio, 4),
        "glare": glare,
        "dark": dark,
        "note": "、".join(notes) if notes else "图片质量基本可用",
    }


def image_to_data_url(image: Image.Image, max_edge: int, quality: int) -> str:
    if max_edge < 1 or not 1 <= quality <= 100:
        raise ValueError("Image max edge must be positive and JPEG quality must be in 1..100.")
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / longest
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_image_data_url(image_path: Path, max_edge: int, quality: int) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image_to_data_url(image, max_edge=max_edge, quality=quality)


def parse_detail_crop(value: str, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Parse an x1,y1,x2,y2 crop on the original-image 0..1000 grid."""
    if not value or value.strip().lower() in {"off", "none", "关闭"}:
        return None
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive.")
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        raise ValueError("DETAIL_CROP must be auto or x1,y1,x2,y2 on the 0..1000 grid.")
    if len(values) != 4 or not all(math.isfinite(part) for part in values):
        raise ValueError("DETAIL_CROP must be auto or x1,y1,x2,y2 on the 0..1000 grid.")
    if not all(0 <= part <= 1000 for part in values):
        raise ValueError("DETAIL_CROP coordinates must be within 0..1000.")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("DETAIL_CROP coordinates must satisfy x2>x1 and y2>y1.")
    left = max(0, min(width - 1, math.floor(x1 * width / 1000)))
    top = max(0, min(height - 1, math.floor(y1 * height / 1000)))
    right = max(left + 1, min(width, math.ceil(x2 * width / 1000)))
    bottom = max(top + 1, min(height, math.ceil(y2 * height / 1000)))
    return left, top, right, bottom


def select_auto_detail_crop(image_path: Path) -> tuple[int, int, int, int]:
    """Select one high-detail region without adding a model call.

    This is only a routing heuristic. The model must still verify the crop
    against the full image, and callers can override it with an explicit box.
    """
    with Image.open(image_path) as source:
        return select_auto_crop_from_image(ImageOps.exif_transpose(source))


def select_auto_crop_from_image(image: Image.Image) -> tuple[int, int, int, int]:
    image = image.convert("L")
    width, height = image.size
    crop_width = max(1, int(width * 0.62))
    crop_height = max(1, int(height * 0.62))
    sample = image.resize((min(width, 480), min(height, 480)))
    edge = sample.filter(ImageFilter.FIND_EDGES)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for center_x in (0.25, 0.5, 0.75):
        for center_y in (0.25, 0.5, 0.75):
            left = max(0, min(width - crop_width, int(width * center_x - crop_width / 2)))
            top = max(0, min(height - crop_height, int(height * center_y - crop_height / 2)))
            right, bottom = left + crop_width, top + crop_height
            sx1 = int(left * sample.width / width)
            sy1 = int(top * sample.height / height)
            sx2 = max(sx1 + 1, int(right * sample.width / width))
            sy2 = max(sy1 + 1, int(bottom * sample.height / height))
            score = ImageStat.Stat(edge.crop((sx1, sy1, sx2, sy2))).mean[0]
            candidates.append((score, (left, top, right, bottom)))
    return max(candidates, key=lambda item: item[0])[1]


def prepare_detail_crop_data_url(
    image_path: Path,
    crop_box: tuple[int, int, int, int],
    max_edge: int,
    quality: int,
) -> str:
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        left, top, right, bottom = crop_box
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError("Detail crop must be nonempty and inside the oriented original image.")
        return image_to_data_url(image.crop(crop_box), max_edge=max_edge, quality=quality)


def crop_box_to_1000(
    crop_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> list[int]:
    width, height = image_size
    left, top, right, bottom = crop_box
    return [
        round(left * 1000 / width),
        round(top * 1000 / height),
        round(right * 1000 / width),
        round(bottom * 1000 / height),
    ]


def original_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def bbox_1000_to_original(
    bbox: Any,
    image_width: int,
    image_height: int,
) -> dict[str, int] | None:
    """Convert model [x1,y1,x2,y2] coordinates on a 0..1000 grid to pixels."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, y1, x2, y2 = (max(0.0, min(1000.0, value)) for value in (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1 or image_width <= 0 or image_height <= 0:
        return None
    left = max(0, min(image_width - 1, math.floor(x1 * image_width / 1000)))
    top = max(0, min(image_height - 1, math.floor(y1 * image_height / 1000)))
    right = max(left + 1, min(image_width, math.ceil(x2 * image_width / 1000)))
    bottom = max(top + 1, min(image_height, math.ceil(y2 * image_height / 1000)))
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
        "image_width": image_width,
        "image_height": image_height,
    }


def normalize_bbox_1000(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    values = [max(0, min(1000, round(value))) for value in values]
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def build_vision_payload(
    image_data_url: str,
    system_prompt: str,
    user_text: str,
    model: str,
    image_detail: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    detail_image_data_url: str | None = None,
    detail_image_detail: str | None = None,
    detail_crop_box_1000: list[int] | None = None,
) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {
            "type": "image_url",
            "image_url": {
                "url": image_data_url,
                "detail": image_detail,
            },
        },
    ]
    if detail_image_data_url:
        crop_hint = (
            f"局部图对应原图0-1000坐标框：{detail_crop_box_1000}。"
            if detail_crop_box_1000
            else "局部图对应原图的一个细节区域。"
        )
        user_content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "下面附加一张原图局部细节图，仅用于确认小部件、装备和文字。"
                        f"{crop_hint}局部图不能替代全图，也不能改变全图中的空间关系；所有对象和违规框坐标仍必须按整张原图输出。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": detail_image_data_url,
                        "detail": detail_image_detail or image_detail,
                    },
                },
            ]
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled" if reasoning_effort == "none" else "enabled"},
    }
    if reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    return payload



def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _escape_inner_quotes(text: str) -> str:
    """Escape quotes that appear inside a string value.

    In valid JSON a string always ends before a structural character, so a
    quote followed by ordinary text is an inner quote the model left unescaped.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    while index < length:
        char = text[index]
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == "\\":
            out.append(text[index : index + 2])
            index += 2
            continue
        if char == '"':
            look = index + 1
            while look < length and text[look] in " \t\r\n":
                look += 1
            following = text[look] if look < length else ""
            if following in ",}][:" or following == "":
                in_string = False
                out.append(char)
            else:
                out.append('\\"')
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _repair_json_text(text: str) -> str:
    """Repair the JSON glitches a vision model occasionally emits."""
    repaired = text.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\uff02", '"').replace("\u2018", "'").replace("\u2019", "'")
    repaired = _escape_inner_quotes(repaired)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _trim_trailing_closers(text: str, max_trim: int = 3) -> list[str]:
    """The model sometimes closes the JSON object one time too many.

    Measured on the 94-image set: a small share of responses end with an extra
    `}` or `]`. Dropping the surplus closers recovers the report without
    spending another model call, which the eight second budget cannot afford.
    """
    out: list[str] = []
    for cut in range(1, max_trim + 1):
        if len(text) <= cut:
            break
        tail = text[-cut:]
        if all(char in "}]" for char in tail):
            out.append(text[:-cut])
    return out


def _balance_json(text: str) -> str:
    """Close braces and brackets the model left open."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
    if in_string:
        text += '"'
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


def _split_top_level_objects(array_text: str) -> list[str]:
    """Split an array body into its top level object literals."""
    chunks: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    start = 0
    for index, char in enumerate(array_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                chunks.append(array_text[start : index + 1])
                start = index + 1
        elif char == "," and depth == 1:
            start = index + 1
    return chunks


def _salvage_issues(text: str) -> dict[str, Any]:
    """Recover individual issue objects from a damaged JSON document."""
    match = re.search(r'"(?:iss|issues)"\s*:\s*\[', text)
    if not match:
        return {}
    body_start = match.end()
    depth = 1
    in_string = False
    escaped = False
    body_end = body_start
    for index in range(body_start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                body_end = index
                break
    body = text[body_start:body_end]
    recovered: list[Any] = []
    for chunk in _split_top_level_objects(body):
        chunk = chunk.strip()
        if not chunk:
            continue
        for candidate in (chunk, _repair_json_text(chunk), _balance_json(_repair_json_text(chunk))):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                recovered.append(parsed)
                break
    if not recovered:
        # 整条数组都没写完：连最后一条也救出来。
        tail = body[len(body.rstrip().rstrip(",")):].strip()
        if not tail:
            tail = body.rsplit("{", 1)[-1] if "{" in body else ""
        if tail:
            partial = _salvage_partial_issue(tail)
            if partial:
                recovered.append(partial)
    if not recovered:
        return {}
    salvaged: dict[str, Any] = {}
    header = re.search(r'"st"\s*:\s*"([^"]*)"', text)
    if header:
        salvaged["scene_type"] = header.group(1)
    salvaged["issues"] = recovered
    salvaged["_salvaged"] = True
    return salvaged


# 一个 issue 真正不能缺的字段：状态、违规项、风险键。
# 只要这三个还在，截断掉的那一条也能当成一条待复核的风险报出来。
_PARTIAL_ISSUE_FIELDS = (
    ("s", "status"),
    ("i", "item"),
    ("key", "risk_key"),
    ("cat", "risk_category"),
    ("t", "target"),
    ("pos", "position"),
    ("e", "evidence"),
    ("cf", "confidence"),
)


def _salvage_partial_issue(fragment: str) -> dict[str, Any] | None:
    """Rebuild one issue from a JSON object the model never finished.

    A response cut short by the token limit loses the closing braces of the
    last issue. The scalars that were already written are still readable, and
    an issue without a status, an item and a risk key is unusable, so only
    those three are required.
    """
    fragment = fragment.strip().lstrip("{").rstrip(",")
    if not fragment:
        return None
    recovered: dict[str, Any] = {}
    for short, canonical in _PARTIAL_ISSUE_FIELDS:
        match = re.search(rf'"{short}"\s*:\s*"([^"]*)"', fragment)
        if match:
            recovered[canonical] = match.group(1)
            continue
        match = re.search(rf'"{short}"\s*:\s*([0-9.]+)', fragment)
        if match:
            recovered[canonical] = float(match.group(1))
    if not recovered.get("item"):
        return None
    if not recovered.get("status"):
        recovered["status"] = "SUSPECTED"
    box = re.search(r'"b"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', fragment)
    if box:
        recovered["bbox_2d_1000"] = [int(box.group(i)) for i in range(1, 5)]
    recovered["_partial"] = True
    return recovered


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    candidates = [cleaned]
    repaired = _repair_json_text(cleaned)
    if repaired != cleaned:
        candidates.append(repaired)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
        candidates.append(_repair_json_text(cleaned[start : end + 1]))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    trimmed = _trim_trailing_closers(cleaned)
    for candidate in (repaired, _balance_json(repaired), _balance_json(_repair_json_text(cleaned))):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    for candidate in trimmed:
        for variant in (candidate, _balance_json(candidate)):
            try:
                data = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    salvaged = _salvage_issues(cleaned)
    if salvaged:
        return salvaged
    raise ValueError("model response was not a usable JSON object")


def ynu(value: Any) -> str:
    if value is None:
        # Missing structured fields must fail closed instead of becoming
        # "uncertain" and accidentally entering a rule branch.
        return "not_applicable"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no", "uncertain", "not_applicable"}:
            return normalized
        if normalized in {"不适用", "无此对象", "未发现", "不存在", "无"}:
            return "not_applicable"
        if normalized in {"true", "是", "有", "打开", "存在"}:
            return "yes"
        if normalized in {"false", "否", "无", "关闭", "不存在"}:
            return "no"
    return "uncertain"


def normalize_issue_meta(value: Any, kind: str) -> str:
    raw = str(value or "").strip().lower()
    if kind == "visibility":
        mapping = {
            "complete": "complete",
            "full": "complete",
            "完整": "complete",
            "清晰可见": "complete",
            "partial": "partial",
            "部分": "partial",
            "局部": "partial",
            "occluded": "occluded",
            "遮挡": "occluded",
            "被遮挡": "occluded",
            "not_visible": "not_visible",
            "not visible": "not_visible",
            "未入镜": "not_visible",
            "未拍到": "not_visible",
            "不可见": "not_visible",
        }
        return mapping.get(raw, "partial")
    if kind == "evidence":
        mapping = {
            "direct": "direct",
            "直接": "direct",
            "清晰": "direct",
            "partial": "partial",
            "部分": "partial",
            "局部": "partial",
            "occluded": "occluded",
            "遮挡": "occluded",
            "被遮挡": "occluded",
            "not_visible": "not_visible",
            "未入镜": "not_visible",
            "未拍到": "not_visible",
            "ambiguous": "ambiguous",
            "不确定": "ambiguous",
            "模糊": "ambiguous",
        }
        return mapping.get(raw, "ambiguous")
    if kind == "citation":
        if raw in {"high", "高", "高把握"}:
            return "high"
        if raw in {"medium", "中", "中等把握"}:
            return "medium"
        if raw in {"low", "低", "低把握"}:
            return "low"
        return "unknown"
    return raw


PERSON_BODY_PARTS = (
    "head",
    "left_hand",
    "right_hand",
    "torso",
    "waist",
    "left_foot",
    "right_foot",
)
PERSON_PPE_ITEMS = ("helmet", "gloves", "reflective_vest", "safety_harness")
HAND_PPE_ITEMS = ("left_glove", "right_glove")
CLAIM_TYPES = {
    "OPEN_CLOSED",
    "PRESENCE_ABSENCE",
    "MATERIAL_SURFACE",
    "SPATIAL_OCCUPANCY",
    "OBJECT_IDENTITY",
    "FOREIGN_OBJECT",
    "PPE",
    "OTHER",
}
PPE_REQUIRED_BODY_PARTS = {
    "helmet": ("head",),
    "gloves": ("left_hand", "right_hand"),
    "reflective_vest": ("torso",),
    "safety_harness": ("torso", "waist"),
}


def normalize_ppe_state(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "worn": "worn",
        "佩戴": "worn",
        "已佩戴": "worn",
        "not_worn": "not_worn",
        "not worn": "not_worn",
        "未佩戴": "not_worn",
        "未戴": "not_worn",
        "improper": "improper",
        "佩戴不规范": "improper",
        "不规范": "improper",
        "not_applicable": "not_applicable",
        "不适用": "not_applicable",
    }
    return aliases.get(raw, "uncertain")


def normalize_elevation_state(value: Any) -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "GROUND": "GROUND_LEVEL",
        "GROUND_LEVEL": "GROUND_LEVEL",
        "地面": "GROUND_LEVEL",
        "地面作业": "GROUND_LEVEL",
        "LOW_OBSTACLE": "LOW_OBSTACLE",
        "低矮障碍物": "LOW_OBSTACLE",
        "低处材料": "LOW_OBSTACLE",
        "ELEVATED": "ELEVATED_WITH_FALL_RISK",
        "ELEVATED_WITH_FALL_RISK": "ELEVATED_WITH_FALL_RISK",
        "高处": "ELEVATED_WITH_FALL_RISK",
        "高处坠落风险": "ELEVATED_WITH_FALL_RISK",
    }
    return aliases.get(raw, "UNCERTAIN")


def normalize_enum(value: Any, allowed: set[str], default: str = "UNCERTAIN") -> str:
    raw = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return raw if raw in allowed else default


def normalize_objects(
    value: Any,
    original_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    raw_objects = value if isinstance(value, list) else []
    objects: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_object in enumerate(raw_objects[:10], start=1):
        if not isinstance(raw_object, dict):
            continue
        requested_id = re.sub(
            r"[^A-Z0-9_-]", "", str(raw_object.get("object_id") or "").upper()
        )
        object_id = requested_id or f"O{index}"
        if object_id in used_ids:
            object_id = f"O{index}"
        used_ids.add(object_id)
        raw_cues = raw_object.get("identity_cues")
        if not isinstance(raw_cues, list):
            raw_cues = []
        bbox = normalize_bbox_1000(raw_object.get("bbox_2d_1000"))
        parent_id = re.sub(
            r"[^A-Z0-9_-]", "", str(raw_object.get("parent_id") or "").upper()
        ) or None
        objects.append(
            {
                "object_id": object_id,
                "object_type": str(raw_object.get("object_type") or "OTHER").strip()[:80],
                "parent_id": parent_id,
                "bbox_2d_1000": bbox,
                "bbox_original_px": (
                    bbox_1000_to_original(bbox, *original_size)
                    if original_size is not None
                    else None
                ),
                "visibility": normalize_issue_meta(
                    raw_object.get("visibility"), "visibility"
                ),
                "identity_cues": [
                    str(cue).strip()[:100]
                    for cue in raw_cues
                    if str(cue).strip()
                ][:3],
                "observed_state": str(raw_object.get("observed_state") or "").strip()[:120],
            }
        )
    return objects


def normalize_persons(
    value: Any,
    original_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    raw_people = value if isinstance(value, list) else []
    people: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw_person in enumerate(raw_people[:8], start=1):
        if not isinstance(raw_person, dict):
            continue
        requested_id = re.sub(r"[^A-Z0-9_-]", "", str(raw_person.get("person_id") or "").upper())
        person_id = requested_id or f"P{index}"
        if person_id in used_ids:
            person_id = f"P{index}"
        used_ids.add(person_id)

        raw_visibility = raw_person.get("body_visibility")
        if not isinstance(raw_visibility, dict):
            raw_visibility = {}
        body_visibility = {
            part: normalize_issue_meta(raw_visibility.get(part), "visibility")
            for part in PERSON_BODY_PARTS
        }
        raw_ppe = raw_person.get("ppe_states")
        if not isinstance(raw_ppe, dict):
            raw_ppe = {}
        raw_hand_ppe = raw_person.get("hand_ppe_states")
        if not isinstance(raw_hand_ppe, dict):
            raw_hand_ppe = {}
        hand_ppe_states = {
            item: normalize_ppe_state(raw_hand_ppe.get(item))
            for item in HAND_PPE_ITEMS
        }
        ppe_states = {
            item: normalize_ppe_state(raw_ppe.get(item))
            for item in PERSON_PPE_ITEMS
        }
        hand_states = tuple(hand_ppe_states.values())
        if ppe_states["gloves"] == "uncertain" and any(
            state != "uncertain" for state in hand_states
        ):
            if any(state == "not_worn" for state in hand_states):
                ppe_states["gloves"] = "not_worn"
            elif any(state == "improper" for state in hand_states):
                ppe_states["gloves"] = "improper"
            elif all(state == "worn" for state in hand_states):
                ppe_states["gloves"] = "worn"
        raw_cues = raw_person.get("spatial_cues")
        if not isinstance(raw_cues, list):
            raw_cues = []
        bbox = normalize_bbox_1000(raw_person.get("bbox_2d_1000"))
        people.append(
            {
                "person_id": person_id,
                "bbox_2d_1000": bbox,
                "bbox_original_px": (
                    bbox_1000_to_original(bbox, *original_size)
                    if original_size is not None
                    else None
                ),
                "body_visibility": body_visibility,
                "hand_ppe_states": hand_ppe_states,
                "ppe_states": ppe_states,
                "action": str(raw_person.get("action") or "").strip()[:100],
                "support_surface": str(raw_person.get("support_surface") or "unknown").strip()[:100],
                "elevation_state": normalize_elevation_state(raw_person.get("elevation_state")),
                "spatial_cues": [
                    str(cue).strip()[:100]
                    for cue in raw_cues
                    if str(cue).strip()
                ][:3],
            }
        )
    return people


def normalize_person_issue_fields(issue: dict[str, Any]) -> dict[str, Any]:
    result = dict(issue)
    assessment_type = str(result.get("assessment_type") or "OTHER").strip().upper()
    if assessment_type not in {
        "PPE", "SPATIAL", "OBJECT_STATE", "MATERIAL_STATE",
        "PRESENCE_ABSENCE", "OTHER",
    }:
        assessment_type = "OTHER"
    result["assessment_type"] = assessment_type

    claim_type = str(result.get("claim_type") or "OTHER").strip().upper()
    result["claim_type"] = claim_type if claim_type in CLAIM_TYPES else "OTHER"

    raw_object_ids = result.get("object_ids")
    if not isinstance(raw_object_ids, list):
        raw_object_ids = []
    result["object_ids"] = [
        re.sub(r"[^A-Z0-9_-]", "", str(object_id).upper())
        for object_id in raw_object_ids
        if str(object_id).strip()
    ][:5]

    raw_person_ids = result.get("person_ids")
    if not isinstance(raw_person_ids, list):
        raw_person_ids = [result.get("person_id")] if result.get("person_id") else []
    result["person_ids"] = [
        re.sub(r"[^A-Z0-9_-]", "", str(person_id).upper())
        for person_id in raw_person_ids
        if str(person_id).strip()
    ][:4]

    ppe_item = str(result.get("ppe_item") or "not_applicable").strip().lower()
    if ppe_item not in {*PERSON_PPE_ITEMS, "other", "not_applicable"}:
        ppe_item = "other"
    result["ppe_item"] = ppe_item

    raw_parts = result.get("focus_body_parts")
    if not isinstance(raw_parts, list):
        raw_parts = []
    result["focus_body_parts"] = [
        str(part).strip().lower()
        for part in raw_parts
        if str(part).strip().lower() in PERSON_BODY_PARTS
    ]

    raw_spatial = result.get("spatial_relation")
    if not isinstance(raw_spatial, dict):
        raw_spatial = {}
    raw_spatial_cues = raw_spatial.get("cues")
    if not isinstance(raw_spatial_cues, list):
        raw_spatial_cues = []
    both_visible_raw = raw_spatial.get("both_anchors_visible", False)
    both_visible = (
        both_visible_raw
        if isinstance(both_visible_raw, bool)
        else str(both_visible_raw).strip().lower() in {"true", "yes", "1", "是"}
    )
    result["spatial_relation"] = {
        "subject_anchor": str(raw_spatial.get("subject_anchor") or "").strip()[:80],
        "reference_object": str(raw_spatial.get("reference_object") or "").strip()[:80],
        "reference_anchor": str(raw_spatial.get("reference_anchor") or "").strip()[:80],
        "both_anchors_visible": both_visible,
        "elevation_state": normalize_elevation_state(raw_spatial.get("elevation_state")),
        "reference_state": normalize_enum(
            raw_spatial.get("reference_state"),
            {"STATIC", "SUSPENDED", "MOVING", "UNCERTAIN"},
        ),
        "person_zone": normalize_enum(
            raw_spatial.get("person_zone"),
            {"UNDER_LOAD", "SWING_PATH", "ADJACENT", "OUTSIDE", "UNCERTAIN"},
        ),
        "cues": [
            str(cue).strip()[:100]
            for cue in raw_spatial_cues
            if str(cue).strip()
        ][:3],
    }

    raw_validation = result.get("claim_validation")
    if not isinstance(raw_validation, dict):
        raw_validation = {}

    def as_bool(key: str) -> bool:
        value = raw_validation.get(key, False)
        return value if isinstance(value, bool) else str(value).strip().lower() in {
            "true", "yes", "1", "是",
        }

    raw_positive_cues = raw_validation.get("positive_cues")
    if not isinstance(raw_positive_cues, list):
        raw_positive_cues = []
    result["claim_validation"] = {
        "target_identity_confirmed": as_bool("target_identity_confirmed"),
        "component_boundary_complete": as_bool("component_boundary_complete"),
        "inspection_region_complete": as_bool("inspection_region_complete"),
        "positive_cues": [
            str(cue).strip()[:100]
            for cue in raw_positive_cues
            if str(cue).strip()
        ][:3],
        "alternative_explanation": str(
            raw_validation.get("alternative_explanation") or ""
        ).strip()[:120],
        "alternative_excluded": as_bool("alternative_excluded"),
        "subject_anchor": str(raw_validation.get("subject_anchor") or "").strip()[:80],
        "reference_anchor": str(raw_validation.get("reference_anchor") or "").strip()[:80],
        "relation": normalize_enum(
            raw_validation.get("relation"),
            {"OVERLAP", "BLOCKING", "CONTACT", "ADJACENT", "SEPARATE", "UNCERTAIN"},
        ),
    }
    return result


def apply_claim_evidence_gate(
    issue: dict[str, Any],
    status: str,
    objects_by_id: dict[str, dict[str, Any]] | None,
) -> str:
    """Require structured visual evidence for commonly confused state claims."""
    claim_type = issue.get("claim_type", "OTHER")
    if claim_type in {"OTHER", "PPE"}:
        return status

    objects_by_id = objects_by_id or {}
    object_ids = issue.get("object_ids", [])
    referenced_objects = [
        objects_by_id[object_id]
        for object_id in object_ids
        if object_id in objects_by_id
    ]
    validation = issue.get("claim_validation", {})
    if not object_ids or len(referenced_objects) != len(object_ids):
        return "NOT_ASSESSABLE"
    if not validation.get("target_identity_confirmed"):
        return "NOT_ASSESSABLE"
    if len(validation.get("positive_cues", [])) < 2:
        return "NOT_ASSESSABLE"
    if not validation.get("alternative_explanation") or not validation.get("alternative_excluded"):
        return "NOT_ASSESSABLE"

    if claim_type == "OPEN_CLOSED" and not validation.get("component_boundary_complete"):
        return "NOT_ASSESSABLE"
    if claim_type == "PRESENCE_ABSENCE" and not validation.get("inspection_region_complete"):
        return "NOT_ASSESSABLE"
    if claim_type == "FOREIGN_OBJECT" and not validation.get("component_boundary_complete"):
        return "NOT_ASSESSABLE"
    if claim_type == "SPATIAL_OCCUPANCY":
        if not validation.get("subject_anchor") or not validation.get("reference_anchor"):
            return "NOT_ASSESSABLE"
        if validation.get("relation") not in {"OVERLAP", "BLOCKING", "CONTACT"}:
            return "NOT_ASSESSABLE"
    return status


def apply_person_issue_gate(
    issue: dict[str, Any],
    status: str,
    people_by_id: dict[str, dict[str, Any]] | None,
) -> str:
    """Downgrade unsupported PPE and spatial claims without deciding the risk."""
    people_by_id = people_by_id or {}
    assessment_type = issue.get("assessment_type")
    person_ids = issue.get("person_ids", [])
    referenced_people = [people_by_id[person_id] for person_id in person_ids if person_id in people_by_id]

    if assessment_type in {"PPE", "SPATIAL"} and (
        not person_ids or len(referenced_people) != len(person_ids)
    ):
        return "NOT_ASSESSABLE"

    if assessment_type == "PPE":
        ppe_item = str(issue.get("ppe_item") or "other")
        required_parts = PPE_REQUIRED_BODY_PARTS.get(
            ppe_item,
            tuple(issue.get("focus_body_parts") or ()),
        )
        if not required_parts:
            return "SUSPECTED"
        for person in referenced_people:
            visibility = person.get("body_visibility", {})
            if any(visibility.get(part) != "complete" for part in required_parts):
                return "NOT_ASSESSABLE"
            if status == "CLEAR" and ppe_item in PERSON_PPE_ITEMS:
                if person.get("ppe_states", {}).get(ppe_item) not in {"not_worn", "improper"}:
                    return "NOT_ASSESSABLE"
                if ppe_item == "gloves":
                    hand_states = person.get("hand_ppe_states", {})
                    known_hand_states = [
                        hand_states.get(item)
                        for item in HAND_PPE_ITEMS
                        if hand_states.get(item) != "uncertain"
                    ]
                    if known_hand_states and len(known_hand_states) != len(HAND_PPE_ITEMS):
                        return "NOT_ASSESSABLE"

    if assessment_type == "SPATIAL" and status == "CLEAR":
        spatial = issue.get("spatial_relation", {})
        if not spatial.get("both_anchors_visible") or len(spatial.get("cues", [])) < 2:
            return "NOT_ASSESSABLE"
        focus_parts = tuple(issue.get("focus_body_parts") or ())
        if not focus_parts or any(
            person.get("body_visibility", {}).get(part) != "complete"
            for person in referenced_people
            for part in focus_parts
        ):
            return "NOT_ASSESSABLE"
        text = f"{issue.get('risk_key', '')} {issue.get('item', '')}"
        high_place_claim = (
            issue.get("risk_category") == "WORK_AT_HEIGHT"
            or re.search(r"HIGH|HEIGHT|CLIMB|ELEVAT|高处|攀爬|登高", text, re.IGNORECASE)
        )
        if high_place_claim:
            if spatial.get("elevation_state") != "ELEVATED_WITH_FALL_RISK":
                return "NOT_ASSESSABLE"
            if any(
                person.get("elevation_state") != "ELEVATED_WITH_FALL_RISK"
                for person in referenced_people
            ):
                return "NOT_ASSESSABLE"
        lifting_claim = (
            issue.get("risk_category") == "LIFTING_OPERATIONS"
            or re.search(r"SUSPENDED_LOAD|LIFTING_DANGER|吊物下方|吊装危险", text, re.IGNORECASE)
        )
        if lifting_claim and (
            spatial.get("reference_state") not in {"SUSPENDED", "MOVING"}
            or spatial.get("person_zone") not in {"UNDER_LOAD", "SWING_PATH"}
        ):
            return "NOT_ASSESSABLE"
    return status


def apply_static_image_scope_gate(
    issue: dict[str, Any],
    status: str,
    people_by_id: dict[str, dict[str, Any]] | None = None,
    objects_by_id: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Downgrade claims that require design data, records, tests, or task context."""
    if status != "CLEAR":
        return status

    key = str(issue.get("risk_key") or "").upper()
    text = " ".join(
        str(issue.get(field) or "")
        for field in ("item", "target", "evidence")
    )
    nonvisual_key = re.search(
        r"QUALIFICATION|LICENSE|CERTIFICATE|PERMIT|"
        r"INSPECTION_RECORD|MAINTENANCE_RECORD|TEST_RECORD|"
        r"DESIGN_COMPLIANCE|LOAD_CAPACITY|OVERLOAD|"
        r"UNSUPPORTED_PIT|PIT_WALL_UNSUPPORTED",
        key,
    )
    nonvisual_text = re.search(
        r"无资质|未持证|无证|未审批|无方案|未检测|无检测记录|"
        r"未维护|无维护记录|超过设计荷载|承载力不足|"
        r"基坑坑壁未支护|坑壁无支护|边坡未支护",
        text,
    )
    if nonvisual_key or nonvisual_text:
        return "SUSPECTED"

    people_by_id = people_by_id or {}
    objects_by_id = objects_by_id or {}
    person_context = " ".join(
        str(people_by_id[person_id].get("action") or "")
        for person_id in issue.get("person_ids", [])
        if person_id in people_by_id
    )
    object_context = " ".join(
        f"{objects_by_id[object_id].get('object_type', '')} "
        f"{objects_by_id[object_id].get('observed_state', '')}"
        for object_id in issue.get("object_ids", [])
        if object_id in objects_by_id
    )
    independent_context = f"{person_context} {object_context}".strip()

    if issue.get("assessment_type") == "PPE" and issue.get("ppe_item") == "gloves":
        task_hazard_visible = re.search(
            r"检修|维修|接线|带电|焊接|切割|打磨|化学|腐蚀|高温|"
            r"锐器|锋利|搬运粗糙|有毒|低温",
            independent_context,
        )
        if not task_hazard_visible:
            return "SUSPECTED"
    if issue.get("assessment_type") == "PPE" and issue.get("ppe_item") == "reflective_vest":
        reflective_property_claim = re.search(
            r"反光条|反光带|反光性能|颜色不符合|颜色不规范",
            f"{key} {text}",
        )
        if reflective_property_claim:
            return "NOT_ASSESSABLE"
        traffic_task_visible = re.search(
            r"交通指挥|车辆引导|道路施工|交通疏导|车辆通道",
            independent_context,
        )
        moving_vehicle_visible = re.search(
            r"行驶中|移动中|倒车|车辆通道|施工道路",
            object_context,
        )
        if not (traffic_task_visible and moving_vehicle_visible):
            return "SUSPECTED"
    return status


def format_model_regulation_candidate(candidate: dict[str, Any]) -> str | None:
    """Format an auditable candidate without presenting it as verified law."""
    required = ("standard_name", "standard_code", "article", "clause_summary")
    if not all(str(candidate.get(key) or "").strip() for key in required):
        return None
    return (
        "模型候选（未经过本地法规目录核验，不可作为正式引用）："
        f"《{candidate['standard_name']}》{candidate['standard_code']} "
        f"{candidate['article']}：{candidate['clause_summary']}"
    )


def normalize_model_regulation(value: Any) -> dict[str, Any]:
    """Normalize the model's unverified regulation candidate for audit."""
    regulation = value if isinstance(value, dict) else {}
    status = str(regulation.get("match_status") or "NEEDS_VERIFICATION").strip().upper()
    if status not in {"MATCHED", "NEEDS_VERIFICATION"}:
        status = "NEEDS_VERIFICATION"
    normalized = {
        "standard_name": str(regulation.get("standard_name") or "").strip()[:120],
        "standard_code": str(regulation.get("standard_code") or "").strip()[:60],
        "article": str(regulation.get("article") or "").strip()[:60],
        "clause_summary": str(regulation.get("clause_summary") or "").strip()[:240],
        "match_reason": str(regulation.get("match_reason") or "").strip()[:160],
        "match_status": status,
        "source": "model_internal_knowledge",
        "verification_required": True,
    }
    if not all(
        normalized[key]
        for key in ("standard_name", "standard_code", "article", "clause_summary")
    ):
        normalized["match_status"] = "NEEDS_VERIFICATION"
    return normalized


def _backfill_regulation_from_catalog(candidate: dict[str, Any]) -> None:
    """Fill standard name and clause text from the local catalog.

    Compact mode asks the model for standard_code and article only; the local
    catalog is the authority for the name, the clause text and its currency.
    """
    code = _catalog_code(candidate.get("standard_code"))
    article = _catalog_article(candidate.get("article"))
    if not code or not article:
        return
    for record in VERIFIED_REGULATIONS.values():
        if (
            _catalog_code(record.get("standard_code")) == code
            and _catalog_article(record.get("article")) == article
        ):
            candidate["standard_name"] = record["standard_name"]
            candidate["clause_summary"] = record["clause_text"]
            candidate.setdefault("match_reason", "条款由本地已核验目录回填")
            if record.get("version_status") == "current_per_project_rulepack":
                candidate["match_status"] = "MATCHED"
            return


def _catalog_code(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _catalog_article(value: Any) -> str:
    return re.sub(r"[^0-9]", "", str(value or ""))


def strict_issue_quality_gate(
    issue: dict[str, Any],
    scene_type: str | None = None,
    people_by_id: dict[str, dict[str, Any]] | None = None,
    objects_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate model evidence metadata without deciding the safety domain itself."""
    result = dict(issue)
    item = str(result.get("item", "安全风险线索"))
    evidence = str(result.get("evidence", "图片可见风险线索，需复核"))
    visibility = normalize_issue_meta(result.get("target_visibility"), "visibility")
    evidence_level = normalize_issue_meta(result.get("evidence_level"), "evidence")

    status = str(result.get("status", "SUSPECTED")).upper()
    if status == "CLEAR" and (visibility != "complete" or evidence_level != "direct"):
        status = "SUSPECTED"
    uncertainty_text = f"{item} {evidence}"
    if status == "CLEAR" and re.search(
        r"无法确认|不能确认|未能确认|难以确认|无法判断|不能判断|看不清|不确定",
        uncertainty_text,
    ):
        status = "NOT_ASSESSABLE"
    elif status == "CLEAR" and re.search(r"疑似|可能|或许|似乎", uncertainty_text):
        status = "SUSPECTED"
    if visibility in {"occluded", "not_visible"} or evidence_level in {"occluded", "not_visible"}:
        status = "NOT_ASSESSABLE"
    elif evidence_level == "ambiguous":
        status = "NOT_ASSESSABLE"

    result = normalize_person_issue_fields(result)
    result["item"] = item[:80]
    result["risk_key"] = str(result.get("risk_key") or item)[:80]
    result["risk_category"] = normalize_scene(result.get("risk_category") or scene_type)
    result["position"] = str(result.get("position", ""))[:40]
    result["evidence"] = evidence[:160]
    cues = result.get("visual_cues")
    if not isinstance(cues, list):
        cues = []
    result["visual_cues"] = [str(cue).strip()[:100] for cue in cues if str(cue).strip()][:3]
    result["target"] = str(result.get("target", "未指定对象"))[:60]
    result["target_visibility"] = visibility
    result["evidence_level"] = evidence_level
    result["citation_confidence"] = normalize_issue_meta(
        result.get("citation_confidence"), "citation"
    )
    candidate = normalize_model_regulation(result.get("regulation"))
    _backfill_regulation_from_catalog(candidate)
    verified, regulation_match = verify_regulation(
        candidate=candidate,
        risk_key=result["risk_key"],
        item_text=result["item"],
        target=result["target"],
        evidence=result["evidence"],
        scene_type=result["risk_category"],
    )
    if verified is not None:
        verified["source"] = "local_verified_catalog"
        verified["verification_required"] = False
    result["model_regulation_candidate"] = candidate
    result["verified_regulation"] = verified
    result["regulation_match"] = regulation_match
    result["model_regulation_candidate_text"] = format_model_regulation_candidate(candidate)
    result["regulation_status"] = (
        "VERIFIED"
        if verified is not None
        else "MODEL_CANDIDATE_UNVERIFIED"
        if result["model_regulation_candidate_text"]
        else "UNMATCHED"
    )
    result["has_verified_regulation"] = verified is not None
    result["regulation_verification_level"] = (
        str(regulation_match.get("method"))
        if isinstance(regulation_match, dict) and regulation_match.get("verified")
        else "none"
    )
    # Compatibility field: consumers that previously read `regulation` now
    # receive only a locally verified canonical clause, never a model guess.
    result["regulation"] = verified
    result["rule"] = format_verified_regulation(verified)
    if verified is None:
        result["citation_confidence"] = "unknown"
    if status == "CLEAR" and len(result["visual_cues"]) < 2:
        status = "SUSPECTED"
    status = apply_claim_evidence_gate(result, status, objects_by_id)
    status = apply_person_issue_gate(result, status, people_by_id)
    status = apply_static_image_scope_gate(
        result,
        status,
        people_by_id=people_by_id,
        objects_by_id=objects_by_id,
    )
    if status == "CLEAR" and regulation_match.get("method") == "keyword":
        status = "SUSPECTED"
        result["review_reason"] = "仅通过多关键词命中法规，尚不足以作为明确违规依据，需人工确认条款适用性。"
    if status == "CLEAR" and verified is None:
        status = "SUSPECTED"
        result["review_reason"] = "风险有视觉线索，但尚无本地已核验的适用条款，不输出为明确违规。"
    elif status != "CLEAR":
        result["review_reason"] = str(result.get("review_reason") or "证据或条款适用性不足，需人工复核。")
    else:
        result["review_reason"] = None
    result["status"] = status if status in {"CLEAR", "SUSPECTED", "NOT_ASSESSABLE"} else "SUSPECTED"
    result["needs_review"] = bool(
        result.get("needs_review", result["status"] != "CLEAR")
        or result["status"] != "CLEAR"
        or verified is None
    )
    return result


def normalize_fact_result(
    data: dict[str, Any],
    scene_override: str | None = None,
    original_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    raw_model_scene = str(data.get("scene_type") or data.get("scene") or "").strip()
    model_scene = normalize_scene(raw_model_scene) if raw_model_scene else ""
    # 调用方给定的场景是提示词与清单的依据（实测模型在有提示时与调用方一致 92/93），
    # 报告里的场景应以调用方为准，模型自判单独记录，不要覆盖。
    detected_scene = normalize_scene(scene_override) if scene_override else (model_scene or "AUTO")
    facts = data.get("facts")
    if not isinstance(facts, dict):
        facts = {}
    profile = get_scene_profile(detected_scene)
    field_names = tuple(profile.get("fields", {}).keys())
    normalized_facts = {
        key: ynu(facts.get(key))
        for key in field_names
    }
    persons = normalize_persons(data.get("persons"), original_size=original_size)
    people_by_id = {person["person_id"]: person for person in persons}
    objects = normalize_objects(data.get("objects"), original_size=original_size)
    objects_by_id = {item["object_id"]: item for item in objects}
    issues = data.get("issues")
    if not isinstance(issues, list):
        issues = []
    normalized_issues: list[dict[str, Any]] = []
    for issue in issues[: max(5, max_risk_items())]:
        if not isinstance(issue, dict):
            continue
        status = str(issue.get("status", "SUSPECTED")).upper()
        if status not in {"CLEAR", "SUSPECTED", "NOT_ASSESSABLE"}:
            status = "SUSPECTED"
        try:
            confidence = float(issue.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        normalized_issue = strict_issue_quality_gate(
            {
                "status": status,
                "item": str(issue.get("item", "安全风险线索"))[:80],
                "risk_category": issue.get("risk_category", detected_scene),
                "risk_key": issue.get("risk_key"),
                "assessment_type": issue.get("assessment_type", "OTHER"),
                "claim_type": issue.get("claim_type", "OTHER"),
                "object_ids": issue.get("object_ids"),
                "person_ids": issue.get("person_ids"),
                "person_id": issue.get("person_id"),
                "ppe_item": issue.get("ppe_item", "not_applicable"),
                "focus_body_parts": issue.get("focus_body_parts"),
                "spatial_relation": issue.get("spatial_relation"),
                "claim_validation": issue.get("claim_validation"),
                "target": str(issue.get("target", "未指定对象"))[:60],
                "position": str(issue.get("position", ""))[:40],
                "evidence": str(issue.get("evidence", "图片可见风险线索，需复核"))[:160],
                "visual_cues": issue.get("visual_cues"),
                "regulation": issue.get("regulation"),
                "confidence": max(0.0, min(1.0, confidence)),
                # Missing evidence metadata must fail closed. The model remains
                # the safety judge, but an omitted visibility/evidence field
                # cannot silently turn a claim into a clear violation.
                "target_visibility": issue.get("target_visibility", "partial"),
                "evidence_level": issue.get("evidence_level", "ambiguous"),
                "citation_confidence": issue.get("citation_confidence", "unknown"),
                "needs_review": bool(issue.get("needs_review", status != "CLEAR")),
            },
            scene_type=detected_scene,
            people_by_id=people_by_id,
            objects_by_id=objects_by_id,
        )
        normalized_bbox = normalize_bbox_1000(issue.get("bbox_2d_1000"))
        normalized_issue["bbox_2d_1000"] = normalized_bbox
        normalized_issue["bbox_original_px"] = (
            bbox_1000_to_original(normalized_bbox, *original_size)
            if original_size is not None
            else None
        )
        normalized_issues.append(normalized_issue)

    image_quality = data.get("image_quality") if isinstance(data.get("image_quality"), dict) else {}
    try:
        scene_confidence = max(0.0, min(1.0, float(data.get("scene_confidence", 0.0))))
    except (TypeError, ValueError):
        scene_confidence = 0.0
    raw_scene_evidence = data.get("scene_evidence")
    if not isinstance(raw_scene_evidence, list):
        raw_scene_evidence = []
    return {
        "scene_type": detected_scene,
        "scene": profile.get("label", "未识别"),
        "scene_hint": scene_override,
        "model_detected_scene": model_scene or None,
        "model_scene_confidence": scene_confidence,
        "scene_confidence": scene_confidence,
        "scene_evidence": [
            str(cue).strip()[:100]
            for cue in raw_scene_evidence
            if str(cue).strip()
        ][:3],
        "image_quality": image_quality,
        "facts": normalized_facts,
        "objects": objects,
        "persons": persons,
        "issues": normalized_issues,
    }


def issue_status_zh(status: str) -> str:
    return {
        "CLEAR": "明确违规",
        "SUSPECTED": "疑似违规",
        "NOT_ASSESSABLE": "不可判断",
    }.get(str(status).upper(), "疑似违规")



def model_judged_report(
    facts_data: dict[str, Any],
    local_quality: dict[str, Any],
    elapsed: float | None = None,
    scene: str | None = None,
    prompt: str = "",
) -> str:
    scene_key = normalize_scene(scene or facts_data.get("scene_type"))
    profile = get_scene_profile(scene_key)
    source_issues = facts_data.get("issues", [])
    issues = list(source_issues) if isinstance(source_issues, list) else []
    notes: list[str] = []

    quality_note = local_quality.get("note", "图片质量基本可用")
    if local_quality.get("clarity") != "clear" or local_quality.get("glare") or local_quality.get("dark"):
        notes.append(f"图像质量提示：{quality_note}，相关空间状态只作为疑似或不可判断。")

    lines = [
        "报告格式版本：1.3",
        "检测模型：DeepSeek Flash（单模型单次视觉检测）",
        f"检查场景：{profile['label']}",
        "规范匹配方式：DeepSeek给出候选，本地已核验现行法规目录负责核验与规范化",
        f"原图尺寸：{int(local_quality.get('width', 0))} × {int(local_quality.get('height', 0))} 像素",
        "",
        "检测结果：以下为模型基于图片可见证据给出的安全风险初筛结果。",
        "",
    ]
    if not issues:
        lines.extend(
            [
                "未发现可直接确认的明显违规。",
                "说明：该结论仅表示本次单张图片中未识别到证据充分的违规项，不代表现场整体合规。",
                "",
            ]
        )
    for idx, issue in enumerate(issues[:8], start=1):
        status_code = str(issue.get("status", "SUSPECTED")).upper()
        status = issue_status_zh(status_code)
        lines.append(f"{idx}. {status}：{issue.get('item', '安全风险线索')}")
        person_ids = issue.get("person_ids")
        if isinstance(person_ids, list) and person_ids:
            lines.append(f"• 关联人员：{', '.join(str(person_id) for person_id in person_ids)}")
        position = str(issue.get("position") or issue.get("target") or "").strip()
        if position:
            lines.append(f"• 目标位置：{position}")
        lines.append(f"• 现象描述：{issue.get('evidence', '图片可见风险线索，需复核')}")
        regulation_label = (
            "违反的具体安全条例"
            if status_code == "CLEAR"
            else "可能涉及的安全条例（尚未确认违反）"
        )
        lines.append(
            f"• {regulation_label}："
            f"{issue.get('rule', format_verified_regulation(None))}"
        )
        if issue.get("verified_regulation") is None and issue.get("model_regulation_candidate_text"):
            lines.append(f"• {issue['model_regulation_candidate_text']}")
        regulation_match = issue.get("regulation_match")
        if isinstance(regulation_match, dict):
            lines.append(
                "• 法规核验："
                f"{regulation_match.get('method', 'none')}，"
                f"本地目录匹配分={float(regulation_match.get('confidence', 0.0)):.2f}（不是法规真实性概率）；"
                f"{regulation_match.get('note', '')}"
            )
        if isinstance(issue.get("verified_regulation"), dict):
            lines.append(
                "• 法规来源范围：本地法规目录匹配，非实时联网核验；"
                "请结合条文适用条件和官方现行版本复核。"
            )
        bbox = issue.get("bbox_original_px")
        if isinstance(bbox, dict):
            lines.append(
                "• 违规范围框（原图像素，左上角原点）："
                f"x={bbox['x']}, y={bbox['y']}, "
                f"width={bbox['width']}, height={bbox['height']}"
            )
        else:
            lines.append("• 违规范围框：无法从当前图片证据中可靠定位（bbox=null）")
        lines.append(
            f"• 单模型判断置信度：{float(issue.get('confidence', 0.0)):.2f}"
        )
        lines.append("")

    if notes:
        lines.append("系统复核提示：")
        lines.extend(f"- {note}" for note in notes)

    if elapsed is not None:
        lines.append("")
        lines.append(f"流程耗时：{elapsed:.2f} 秒")
    return "\n".join(lines).strip()


RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ModelBudgetExhausted(httpx.TimeoutException):
    """The request budget is gone, so another attempt cannot help."""


class ModelNetworkError(RuntimeError):
    """Transient network or upstream failure that survived every allowed attempt."""


class ModelQuotaError(RuntimeError):
    """The provider rejected the call for billing or quota reasons.

    This is not a detection failure and must not be recorded as one: the caller
    has to top up and try again, otherwise the image would be lost.
    """


def is_transient_model_error(exc: BaseException) -> bool:
    """Whether the failure is worth another attempt with the leftover budget.

    DNS failures, refused connections, dropped sockets and overloaded upstreams
    are transient. A malformed request, a bad key or a refused body is not.
    """
    if isinstance(exc, ModelBudgetExhausted):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError,
                        httpx.WriteError, httpx.RemoteProtocolError, httpx.PoolTimeout,
                        httpx.TimeoutException)):
        return True
    if isinstance(exc, (socket.gaierror, ConnectionError)):
        return True
    return False


def call_deepseek(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_attempts: int | None = None,
    retry_min_budget: float | None = None,
) -> tuple[str, int]:
    """Send the request, retrying transient failures inside the time budget.

    The budget is shared by all attempts. A failure that consumed the whole
    budget (a slow upstream) is never retried, while a failure that returned
    early (a DNS lookup that never reached the server) leaves enough budget for
    another attempt.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if max_attempts is None:
        max_attempts = env_int("MODEL_MAX_ATTEMPTS", 3)
    if retry_min_budget is None:
        retry_min_budget = env_float("MODEL_RETRY_MIN_BUDGET", 1.0)
    max_attempts = max(1, int(max_attempts))
    deadline = time.monotonic() + max(0.0, timeout_seconds)

    connect_cap = env_float("MODEL_CONNECT_TIMEOUT", 2.0)

    async def request(budget: float) -> dict[str, Any]:
        # 连接阶段失败（DNS 解析、拒绝连接）要尽快暴露，才有机会用剩余预算重试
        timeout = httpx.Timeout(budget, connect=max(0.3, min(connect_cap, budget)))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def request_with_deadline(budget: float) -> dict[str, Any]:
        # httpx phase timeouts alone do not cap connect + upload + response time.
        try:
            return await asyncio.wait_for(request(budget), timeout=budget)
        except asyncio.TimeoutError as exc:
            raise ModelBudgetExhausted("DeepSeek request exhausted its total time budget.") from exc

    attempts = 0
    last_error: BaseException | None = None
    while True:
        attempts += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelBudgetExhausted("DeepSeek request exhausted its total time budget.") from last_error
        try:
            data = asyncio.run(request_with_deadline(remaining))
            break
        except ModelBudgetExhausted:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {402, 429}:
                raise ModelQuotaError(
                    f"DeepSeek rejected the request with HTTP {status}: "
                    f"{exc.response.text[:160]}"
                ) from exc
            if not is_transient_model_error(exc):
                raise
            last_error = exc
            leftover = deadline - time.monotonic()
            if attempts >= max_attempts or leftover < retry_min_budget:
                raise ModelNetworkError(
                    f"DeepSeek request failed with a transient error after {attempts} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(min(0.2 * attempts, max(0.0, leftover - retry_min_budget)))
        except Exception as exc:  # noqa: BLE001 - classified right below
            last_error = exc
            if not is_transient_model_error(exc):
                raise
            leftover = deadline - time.monotonic()
            if attempts >= max_attempts or leftover < retry_min_budget:
                if isinstance(exc, httpx.TimeoutException) and not isinstance(
                    exc, (httpx.ConnectTimeout, httpx.PoolTimeout)
                ):
                    # 读超时：预算已经用完，不是网络层问题
                    raise ModelBudgetExhausted(
                        "DeepSeek request exhausted its total time budget."
                    ) from exc
                raise ModelNetworkError(
                    f"DeepSeek request failed with a transient error after {attempts} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(min(0.2 * attempts, max(0.0, leftover - retry_min_budget)))

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip(), attempts

        finish_reason = choice.get("finish_reason", "unknown")
        if message.get("reasoning_content"):
            raise RuntimeError(
                "模型返回了思考内容，但没有返回最终检测报告。"
                f"finish_reason={finish_reason}。"
                "请增大 MAX_TOKENS，或将 REASONING_EFFORT 改为 low。"
            )
        raise RuntimeError(
            "模型返回的最终检测报告为空。"
            f"finish_reason={finish_reason}。请检查模型名称和思考参数。"
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {data}") from exc



def save_result(output_dir: Path, image_path: Path, content: str, elapsed: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_path = output_dir / f"{timestamp}-{image_path.stem}-result.md"
    result_path.write_text(
        f"# 安全风险快速检测结果\n\n"
        f"- 图片：{image_path}\n"
        f"- 耗时：{elapsed:.2f} 秒\n\n"
        f"{content}\n",
        encoding="utf-8",
    )
    return result_path


def save_json_result(
    output_dir: Path,
    image_path: Path,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_path = output_dir / f"{timestamp}-{image_path.stem}-result.json"
    saved_payload = dict(payload)
    saved_payload["saved_path"] = str(result_path)
    result_path.write_text(
        json.dumps(saved_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use DeepSeek Flash API to check construction safety risks in one image.")
    parser.add_argument("image", help="Path to the construction-site image.")
    parser.add_argument("--scene", default="", help="Known scene, for example: 施工现场配电箱违规 / 高处作业 / 临边防护.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom user prompt.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved result files.")
    parser.add_argument("--no-save", action="store_true", help="Print only, do not save result file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate image preprocessing and config without calling the API.")
    parser.add_argument("--show-timing", action="store_true", help="Print preprocessing, model, parsing, coordinate conversion, and report timing.")
    parser.add_argument(
        "--detail-crop",
        default="",
        help=(
            "Optional second image in the same model call: 'auto' or "
            "x1,y1,x2,y2 on the original-image 0..1000 grid. "
            "Leave empty/off to use the full image only."
        ),
    )
    parser.set_defaults(json_output=True)
    parser.add_argument(
        "--json-output",
        dest="json_output",
        action="store_true",
        help="Print the standard machine-readable JSON result (default).",
    )
    parser.add_argument(
        "--text-output",
        dest="json_output",
        action="store_false",
        help="Print the human-readable Markdown report instead of standard JSON.",
    )
    return parser.parse_args()


def emit_cli_error(args: argparse.Namespace, error_type: str, message: str, elapsed: float = 0.0) -> None:
    if args.json_output:
        print(json.dumps(
            {
                "status": "error",
                "format_version": "1.3",
                "output_format": "structured_json",
                "error": {"type": error_type, "message": message},
                "elapsed_seconds": round(max(0.0, elapsed), 4),
            },
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print(f"ERROR: {message}", file=sys.stderr)



def _find_env_file() -> Path:
    """Locate the .env file.

    safety_check.py lives in src/, so the file next to it does not exist;
    the project's .env sits in the repository root. Search upward so the
    script keeps working wherever it is placed.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return here.with_name(".env")


def main() -> int:
    load_dotenv(_find_env_file())
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    requested_scene = normalize_scene(args.scene)
    scene_override = args.scene.strip() or None
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions").strip()
    timeout_seconds = env_float("REQUEST_TIMEOUT_SECONDS", 8.0)
    max_edge = env_int("MAX_IMAGE_EDGE", 768)
    jpeg_quality = env_int("JPEG_QUALITY", 70)
    image_detail = os.getenv("IMAGE_DETAIL", "low").strip()
    high_detail_min_edge = env_int("IMAGE_DETAIL_HIGH_MIN_EDGE", 1600)
    # 局部图可以用更高的 detail：全图走低分辨率保证全局，局部走高分辨率保证细节
    detail_image_detail = os.getenv("DETAIL_CROP_IMAGE_DETAIL", "").strip() or None
    detail_crop_setting = (
        args.detail_crop.strip()
        or os.getenv("DETAIL_CROP", "off").strip()
        or "off"
    )
    detail_crop_max_edge = env_int("DETAIL_CROP_MAX_EDGE", max_edge)
    max_tokens = env_int("SINGLE_MODEL_MAX_TOKENS", env_int("MAX_TOKENS", 2200))
    temperature = env_float("TEMPERATURE", 0.0)
    reasoning_effort = os.getenv("REASONING_EFFORT", "none").strip().lower()
    if reasoning_effort not in {"none", "low", "high", "max"}:
        emit_cli_error(
            args,
            "invalid_configuration",
            "REASONING_EFFORT must be none, low, high, or max.",
        )
        return 2

    started = time.perf_counter()
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be finite and positive.")
        preprocess_started = time.perf_counter()
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image_size = image.size
            local_quality = analyze_pil_image_quality(image)
            image_data_url = image_to_data_url(image, max_edge=max_edge, quality=jpeg_quality)
            detail_crop_box = (
                select_auto_crop_from_image(image)
                if detail_crop_setting.lower() == "auto"
                else parse_detail_crop(detail_crop_setting, image_size)
            )
            detail_image_data_url = (
                image_to_data_url(
                    image.crop(detail_crop_box), max_edge=detail_crop_max_edge, quality=jpeg_quality,
                )
                if detail_crop_box is not None
                else None
            )
        detail_crop_box_1000 = (
            crop_box_to_1000(detail_crop_box, image_size)
            if detail_crop_box is not None
            else None
        )
        output_mode = os.getenv("OUTPUT_MODE", "compact").strip().lower()
        compact_output = output_mode != "full"
        system_prompt = build_single_vision_system_prompt(requested_scene, compact=compact_output)
        user_parts = []
        if scene_override:
            user_parts.append(f"用户指定场景：{scene_override}")
        user_parts.extend(
            [
                f"原图显示尺寸：宽{image_size[0]}像素，高{image_size[1]}像素。",
                f"图像质量提示：{local_quality.get('note', '图片质量基本可用')}。",
                f"检测要求：{args.prompt.strip() or DEFAULT_PROMPT}",
                "坐标必须按整张原图归一化到0-1000后输出，禁止按压缩输入图的像素坐标输出。",
            ]
        )
        if image_detail == "adaptive":
            # 只有源图大到确实会丢细节时才用高细节瓦片；
            # 服务端对小图用 low 和 high 的图片 token 完全相同，high 是空操作。
            image_detail = "high" if max(image_size) > high_detail_min_edge else "low"

        payload = build_vision_payload(
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            user_text="\n".join(user_parts),
            model=model,
            image_detail=image_detail,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            detail_image_data_url=detail_image_data_url,
            detail_image_detail=detail_image_detail,
            detail_crop_box_1000=detail_crop_box_1000,
        )
        preprocess_elapsed = time.perf_counter() - preprocess_started
        if args.dry_run:
            dry_result = {
                "status": "dry_run_ok",
                "format_version": "1.3",
                "model": model,
                "vision_calls": 1,
                "image_inputs": 2 if detail_image_data_url else 1,
                "input_strategy": "full_plus_detail_crop" if detail_image_data_url else "full_image_only",
                "detail_crop": {
                    "mode": detail_crop_setting if detail_image_data_url else "off",
                    "bbox_2d_1000": detail_crop_box_1000,
                    "bbox_original_px": (
                        {
                            "x": detail_crop_box[0],
                            "y": detail_crop_box[1],
                            "width": detail_crop_box[2] - detail_crop_box[0],
                            "height": detail_crop_box[3] - detail_crop_box[1],
                        }
                        if detail_crop_box is not None
                        else None
                    ),
                },
                "scene": get_scene_profile(requested_scene)["label"],
                "image": {
                    "path": str(image_path),
                    "width": image_size[0],
                    "height": image_size[1],
                },
                "config": {
                    "max_image_edge": max_edge,
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                    "timeout_seconds": timeout_seconds,
                },
                "timings": {"preprocess": round(preprocess_elapsed, 4)},
            }
            if args.json_output:
                print(json.dumps(dry_result, ensure_ascii=False, indent=2))
            else:
                print("DRY RUN OK")
                print(f"image={image_path}")
                print(f"model={model}")
                print("vision_calls=1")
                print(f"image_inputs={dry_result['image_inputs']}")
                print(f"input_strategy={dry_result['input_strategy']}")
                print(f"detail_crop_1000={detail_crop_box_1000}")
                print(f"scene={dry_result['scene']}")
                print(f"original_image_size={image_size[0]}x{image_size[1]}")
                print(f"max_image_edge={max_edge}")
                print(f"reasoning_effort={reasoning_effort}")
                print(f"max_tokens={max_tokens}")
                print(f"timeout={timeout_seconds:.1f}s")
                print(f"preprocess_elapsed={preprocess_elapsed:.3f}s")
            return 0

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key.startswith("sk-你的"):
            emit_cli_error(args, "missing_api_key", "Please set DEEPSEEK_API_KEY in .env first.")
            return 2

        model_timeout = env_float(
            "DEEPSEEK_VISION_TIMEOUT_SECONDS",
            max(1.0, timeout_seconds - 0.55),
        )
        remaining = timeout_seconds - (time.perf_counter() - started) - 0.35
        if remaining <= 0:
            raise httpx.TimeoutException("Image preprocessing exhausted the detection time budget.")
        if not math.isfinite(model_timeout) or model_timeout <= 0:
            raise ValueError("DEEPSEEK_VISION_TIMEOUT_SECONDS must be finite and positive.")
        model_timeout = min(model_timeout, remaining)
        call_started = time.perf_counter()
        raw, model_attempts = call_deepseek(
            payload,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=model_timeout,
        )
        model_elapsed = time.perf_counter() - call_started

        parse_started = time.perf_counter()
        try:
            parsed = extract_json_object(raw)
        except ValueError as exc:
            # 留一段原始响应，偶发解析失败时才能定位，而不是只看到一个异常类型
            raise ValueError(
                f"{exc} | 原始响应尾部: {raw[-240:]!r}"
            ) from exc
        if compact_output:
            parsed = expand_compact_result(parsed)
        detected = normalize_fact_result(
            parsed,
            scene_override=scene_override,
            original_size=image_size,
        )
        detected["image_quality"] = local_quality
        detected["image_detail"] = image_detail
        detected["image_width"] = image_size[0]
        detected["image_height"] = image_size[1]
        parse_elapsed = time.perf_counter() - parse_started

        report_started = time.perf_counter()
        elapsed = time.perf_counter() - started
        report = model_judged_report(
            detected,
            local_quality,
            elapsed=elapsed,
            scene=detected.get("scene_type"),
            prompt=args.prompt,
        )
        report_elapsed = time.perf_counter() - report_started
        timings = {
            "preprocess": round(preprocess_elapsed, 4),
            "model_attempts": model_attempts,
            "deepseek_vision": round(model_elapsed, 4),
            "parse_normalize": round(parse_elapsed, 4),
            "report_format": round(report_elapsed, 4),
            "total": round(time.perf_counter() - started, 4),
        }
        if args.show_timing:
            report += "\n\n时间分配：\n"
            report += f"- 图片预处理与请求构造：{timings['preprocess']:.3f} 秒\n"
            report += f"- DeepSeek Flash 单次图片检测：{timings['deepseek_vision']:.3f} 秒（尝试 {timings['model_attempts']} 次）\n"
            report += f"- JSON解析、法规核验与坐标换算：{timings['parse_normalize']:.3f} 秒\n"
            report += f"- 报告格式化：{timings['report_format']:.3f} 秒\n"
            report += f"- 总耗时：{timings['total']:.3f} 秒"
    except httpx.TimeoutException:
        emit_cli_error(
            args,
            "timeout",
            f"DeepSeek Flash request exceeded the time budget ({timeout_seconds:.1f}s total).",
            time.perf_counter() - started,
        )
        return 1
    except ModelQuotaError as exc:
        emit_cli_error(
            args,
            "insufficient_balance",
            str(exc),
            time.perf_counter() - started,
        )
        return 1
    except ModelNetworkError as exc:
        emit_cli_error(
            args,
            "network_error",
            str(exc),
            time.perf_counter() - started,
        )
        return 1
    except Exception as exc:
        emit_cli_error(
            args,
            "processing_error",
            str(exc),
            time.perf_counter() - started,
        )
        return 1

    elapsed = timings["total"]
    if elapsed > timeout_seconds:
        print(f"WARNING: Total elapsed time exceeded {timeout_seconds:.1f} seconds.", file=sys.stderr)

    unverified_count = sum(
        1
        for issue in detected.get("issues", [])
        if not isinstance(issue.get("regulation_match"), dict)
        or not issue["regulation_match"].get("verified")
    )
    warnings = []
    if unverified_count:
        warnings.append(
            f"{unverified_count}条风险未匹配到本地已核验现行条款，法规引用需人工复核。"
        )
    model_scene = detected.get("model_detected_scene")
    if model_scene and detected.get("scene_type") and model_scene != detected.get("scene_type"):
        warnings.append(
            f"模型自判场景({model_scene})与传入场景({detected.get('scene_type')})不一致，"
            "本次检查按传入场景的清单执行，请核对场景标注是否正确。"
        )
    response = {
        "status": "success",
        "format_version": "1.3",
        "output_format": "structured_json",
        "model": model,
        "vision_calls": 1,
        "image_inputs": 2 if detail_image_data_url else 1,
        "input_strategy": "full_plus_detail_crop" if detail_image_data_url else "full_image_only",
        "detail_crop": {
            "mode": detail_crop_setting if detail_image_data_url else "off",
            "bbox_2d_1000": detail_crop_box_1000,
            "bbox_original_px": (
                {
                    "x": detail_crop_box[0],
                    "y": detail_crop_box[1],
                    "width": detail_crop_box[2] - detail_crop_box[0],
                    "height": detail_crop_box[3] - detail_crop_box[1],
                }
                if detail_crop_box is not None
                else None
            ),
        },
        "scene": detected.get("scene"),
        "scene_type": detected.get("scene_type"),
        "model_detected_scene": detected.get("model_detected_scene"),
        "model_scene_confidence": detected.get("model_scene_confidence"),
        "image_detail": image_detail,
        "detail_crop_image_detail": detail_image_detail,
        "scene_hint": detected.get("scene_hint"),
        "scene_confidence": detected.get("scene_confidence"),
        "scene_evidence": detected.get("scene_evidence", []),
        "image": {
            "width": image_size[0],
            "height": image_size[1],
            "coordinate_origin": "top_left",
            "coordinate_unit": "pixel",
        },
        "elapsed_seconds": elapsed,
        "objects": detected.get("objects", []),
        "persons": detected.get("persons", []),
        "issue_count": len(detected.get("issues", [])),
        "issues": detected.get("issues", []),
        "warnings": warnings,
        "timings": timings,
    }

    if args.json_output:
        if not args.no_save:
            saved_path = save_json_result(output_dir, image_path=image_path, payload=response)
            response["saved_path"] = str(saved_path)
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        if not args.no_save:
            saved_path = save_result(output_dir, image_path=image_path, content=report, elapsed=elapsed)
            print(f"结果已保存：{saved_path}")
        print(f"\n耗时：{elapsed:.2f} 秒\n")
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
