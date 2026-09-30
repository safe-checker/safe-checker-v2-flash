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
import base64
import io
import json
import math
import os
import re
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
    format_verified_regulation,
    verify_regulation,
)


DEFAULT_PROMPT = (
    "请根据图片中清晰可见的施工现场情况，完成一次全面但快速的施工安全检查。"
    "覆盖人员防护、作业行为、空间关系、临边洞口、临时用电、机械设备、吊装、脚手架、"
    "基坑、消防、材料堆放和通道等可能存在的风险；由你根据图片和施工安全知识判断风险，"
    "说明直接证据，并依据施工安全规范知识匹配相关标准和具体条款。"
    "无法确认的不要臆测，写为“疑似违规”或“不可判断”。"
)


def build_single_vision_system_prompt(scene: str | None) -> str:
    scene_key = normalize_scene(scene)
    profile = get_scene_profile(scene_key)
    current_date = time.strftime("%Y年%m月")
    return f"""你是施工现场安全视觉检测模型。根据输入图片独立完成一次全面、快速的安全检查。
检查场景提示：{profile['label']}。该提示只是待验证线索，不能覆盖图片证据。先根据主体设备、结构、铭牌和用途标识识别实际场景；施工升降机/施工电梯的吊笼、层门、导轨、控制面板及“自动开门、禁止倚靠”等标识不得仅因出现电控柜而归为临时用电。用户未指定场景时按综合施工现场检查。

检查要求：先建立精简的 objects 对象清单，只收录安全判断所需的主体设备、关键部件、作业区域、危险源、防护设施以及被风险引用的对象，最多8个；不要收录与安全判断无关的背景物。再检查个人防护、作业行为、空间关系、临边洞口、高处、临时用电、机械、吊装、脚手架、基坑、消防、材料、通道及隔离警示等风险。只依据图片直接可见证据；不得把阴影、反光、模糊、遮挡或画面裁切推断成违规。未入镜不等于缺失，空间关系须有清楚几何证据。对开/闭、干/湿、有/无、内/外、存在/缺失、占用/相邻等状态，必须确认目标身份、部件归属和边界，给出至少2条相互独立的直接视觉线索，并排除一个最合理的相反解释；否则只能 SUSPECTED 或 NOT_ASSESSABLE。静态图片不能证明资质、票证、检测记录、承载力、功能试验或精确距离。

对象与状态协议：
1. 配电箱主箱门、内层面板、检修门、插座盖和插口是不同部件。主箱门打开必须同时看见主门门板发生角度位移以及主箱开口/内部空间；箱体侧面、门缝、圆形插口或打开的插座盖不能作为主箱门打开证据。箱门关闭但未上锁，应判“未锁闭”，不得判“箱门未关闭”。
2. 积水必须看见连续液体表面，并至少具有水面边界、镜面反射、波纹或对周围物体的倒影/遮挡关系中的两项。单一深色区域、土壤色差、阴影或局部反光不是积水。
3. 判断护栏/围栏缺失前，先逐项盘点上横杆、中横杆、立杆、挡脚板或网片；已有横杆不得因角度、遮挡或与背景同色而写成“未见护栏”。只允许指出实际缺失的具体构件。
4. 箱门内侧固定的系统图、接线图、责任牌、标签和说明页属于设备文件，不是箱内杂物。只有独立放置、与设备无固定关系且明确无关的物品才可判为杂物。
5. 空间占用必须分别定位设备操作区和障碍物，证明二者边界发生实际重叠或阻断通道；仅在设备附近、透视上重叠或位于不同深度，不等于占用。

人物优先检查协议：先逐人建立 persons 清单并分配 P1、P2 等编号，再判断人物风险。对每人按“头部→左手→右手→躯干→腰部→左脚→右脚”的顺序逐部位观察。装备必须按正确佩戴部位检查：安全帽对应头部，左/右手套分别对应左/右手，反光背心对应躯干，安全带对应躯干和腰部；颜色相近、手持物、背景材料或局部遮挡不得当成佩戴或未佩戴证据。左手套和右手套必须分别记录；相关部位不完整清晰时不得声称“未佩戴”，被钢筋、工具、身体或画面边缘遮挡的一侧写 uncertain。手套、面罩、护目镜、反光背心等任务型防护用品只有在具体作业动作、独立危险对象和对应危害均清晰可见，且有明确适用条款时才可判缺失；仅因人员在工地、高处或车辆附近不得判定缺少这些装备。白天照片中背心颜色、反光条外观或未出现强反光不能证明反光性能不合格，也不能据此判定穿戴不规范。每条 item 只描述一个确定状态，禁止使用“未佩戴或佩戴不规范”等二选一表述。空间位置必须以身体锚点和参照物锚点判断，优先检查双脚接触面、支撑面、临边/洞口边界、可见高差和前后遮挡关系。抬腿、跨越材料、站在低矮材料堆旁不等于攀爬或高处作业；只有人员明确位于存在可见坠落高差的高处作业面，才能写 ELEVATED_WITH_FALL_RISK。靠近钢丝绳、车辆或静置材料不等于处于吊物下方；只有吊物与支撑面明显分离、确实处于悬空或移动状态，且人员锚点位于吊物垂直投影或回转路径内，才能明确判定吊装危险区人员风险。

每个风险输出一个紧贴可见违规证据的范围框。坐标基于整张输入图片，采用0到1000归一化整数，原点在左上角，格式为 [x1,y1,x2,y2]，其中x向右、y向下，x1/y1为左上角，x2/y2为右下角。范围框应包住实际违规对象/证据，不要框整张图或无关区域。若一条风险涉及彼此分离的多个位置，拆成多个风险项分别定位。无法从图中可靠定位时 bbox_2d_1000 必须为 null，禁止猜坐标；有可定位证据时必须给有效坐标。

法规要求：当前日期为{current_date}。对每个风险，先依据施工安全规范知识给出最可能的中国现行标准候选，优先使用当前仍有效的最新版本，不要在已知存在新版时引用被替代的旧版。输出规范全称、标准编号、具体条款号和条文要点；不能确认时将 match_status 写 NEEDS_VERIFICATION。法规匹配范围不得局限于临时用电，应覆盖图片实际涉及的高处、吊装、脚手架、基坑、机械、消防和个人防护等场景。程序随后只负责将候选与本地已核验现行条款目录比对；模型仍负责识图和安全判断。

risk_category 使用下列最接近的场景键：GENERAL_SITE、TEMPORARY_ELECTRICITY、WORK_AT_HEIGHT、FOUNDATION_PIT、LIFTING_OPERATIONS、TOWER_CRANE、CONSTRUCTION_HOIST、SCAFFOLD_COUPLER_TYPE、SCAFFOLD_DISC_BUCKLE、SCAFFOLD_CANTILEVER、SCAFFOLD_ATTACHED_LIFTING。risk_key 使用稳定、简短、大写英文下划线编码，表达根本风险，不包含位置和后果，例如 PERSON_UNDER_SUSPENDED_LOAD、CABLE_ON_GROUND；没有合适固定名称时自行给出稳定编码。

状态：CLEAR表示违规状态有直接清晰证据，item和evidence中不得出现“疑似、可能、无法确认、无法判断”等不确定措辞；SUSPECTED表示存在可见线索但仍有不确定性；NOT_ASSESSABLE表示图片不能判断，此时item应描述“某状态无法判断”，不要把未确认事实写成确定违规名称。
只输出一个合法JSON对象，不输出Markdown、代码围栏或解释。按严重度最多输出3条最重要风险；所有中文描述保持简短，单条 evidence 不超过60字，每个线索数组最多2项，不重复解释。格式：
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
        image = image.convert("RGB")
        width, height = image.size
        sample = image.copy()
        sample.thumbnail((320, 320))
        gray = sample.convert("L")
        stat = ImageStat.Stat(gray)
        brightness = float(stat.mean[0])
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edge)
        sharpness = float(edge_stat.var[0])
        pixels = list(gray.getdata())
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


def prepare_image_data_url(image_path: Path, max_edge: int, quality: int) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)

        if longest > max_edge:
            scale = max_edge / longest
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": image_detail,
                        },
                    },
                ],
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



def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON response must be an object.")
    return data


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
    detected_scene = normalize_scene(raw_model_scene or scene_override)
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
    for issue in issues[:5]:
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
                f"核验置信度={float(regulation_match.get('confidence', 0.0)):.2f}；"
                f"{regulation_match.get('note', '')}"
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


def call_deepseek(payload: dict[str, Any], api_key: str, base_url: str, timeout_seconds: float) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(timeout_seconds, connect=min(3.0, timeout_seconds))

    with httpx.Client(timeout=timeout) as client:
        response = client.post(base_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

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



def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
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
        preprocess_started = time.perf_counter()
        image_size = original_image_size(image_path)
        local_quality = analyze_image_quality(image_path)
        image_data_url = prepare_image_data_url(image_path, max_edge=max_edge, quality=jpeg_quality)
        system_prompt = build_single_vision_system_prompt(requested_scene)
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
        payload = build_vision_payload(
            image_data_url=image_data_url,
            system_prompt=system_prompt,
            user_text="\n".join(user_parts),
            model=model,
            image_detail=image_detail,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        preprocess_elapsed = time.perf_counter() - preprocess_started
        if args.dry_run:
            dry_result = {
                "status": "dry_run_ok",
                "format_version": "1.3",
                "model": model,
                "vision_calls": 1,
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
        model_timeout = min(model_timeout, max(1.0, timeout_seconds - preprocess_elapsed - 0.35))
        call_started = time.perf_counter()
        raw = call_deepseek(
            payload,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=model_timeout,
        )
        model_elapsed = time.perf_counter() - call_started

        parse_started = time.perf_counter()
        detected = normalize_fact_result(
            extract_json_object(raw),
            scene_override=scene_override,
            original_size=image_size,
        )
        detected["image_quality"] = local_quality
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
            "deepseek_vision": round(model_elapsed, 4),
            "parse_normalize": round(parse_elapsed, 4),
            "report_format": round(report_elapsed, 4),
            "total": round(time.perf_counter() - started, 4),
        }
        if args.show_timing:
            report += "\n\n时间分配：\n"
            report += f"- 图片预处理与请求构造：{timings['preprocess']:.3f} 秒\n"
            report += f"- DeepSeek Flash 单次图片检测：{timings['deepseek_vision']:.3f} 秒\n"
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
    response = {
        "status": "success",
        "format_version": "1.3",
        "output_format": "structured_json",
        "model": model,
        "scene": detected.get("scene"),
        "scene_type": detected.get("scene_type"),
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
