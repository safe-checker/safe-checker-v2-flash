#!/usr/bin/env python3
"""
DeepSeek Flash construction safety quick checker.

DeepSeek Flash makes the visual safety decisions in one model call. Python
preprocesses the image, validates the structured response, resolves verified
regulation keys, converts bounding boxes, and formats the standard report.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps, ImageStat

from scene_rules import (
    get_scene_profile,
    normalize_scene,
    scene_fact_schema,
    scene_prompt_instructions,
)
from verified_regulations import (
    format_regulation,
    regulation_catalog_for_prompt,
    resolve_regulation,
)


class VisionJob(NamedTuple):
    source: str
    model: str
    payload: dict[str, Any]
    api_key: str
    base_url: str
    timeout: float


class VisionCallResult(NamedTuple):
    source: str
    model: str
    raw: str | None
    elapsed: float
    error: str | None = None


MODEL_DISPLAY_NAMES = {
    "deepseek": "DeepSeek Flash",
    "qwen": "Qwen3.8-Flash",
    "glm": "GLM-5.3-Flash",
    "kimi": "Kimi K2.6",
    "doubao": "豆包",
}


def model_display_name(source: str, model: str = "") -> str:
    return MODEL_DISPLAY_NAMES.get(source, model or source)


DEFAULT_PROMPT = (
    "请根据图片中清晰可见的施工现场情况，完成一次全面但快速的施工安全检查。"
    "覆盖人员防护、作业行为、空间关系、临边洞口、临时用电、机械设备、吊装、脚手架、"
    "基坑、消防、材料堆放和通道等可能存在的风险；由你根据图片和施工安全知识判断风险，"
    "说明直接证据和对应规范。无法确认的不要臆测，写为“疑似违规”或“不可判断”。"
)


def build_single_vision_system_prompt(scene: str | None) -> str:
    scene_key = normalize_scene(scene)
    profile = get_scene_profile(scene_key)
    return f"""你是施工现场安全视觉检测模型。根据输入图片独立完成一次全面、快速的安全检查。
检查场景提示：{profile['label']}。场景只提供检查方向，不限制其他可见安全风险；用户未指定场景时按综合施工现场检查。

检查要求：先识别图中人员、设备、作业区域、危险源和防护设施，再检查个人防护、作业行为、空间关系、临边洞口、高处、临时用电、机械、吊装、脚手架、基坑、消防、材料、通道及隔离警示等风险。只依据图片直接可见证据；不得把阴影、反光、模糊、遮挡或画面裁切推断成违规。未入镜不等于缺失，空间关系须有清楚几何证据。组合设备的部件分别判断。静态图片不能证明资质、票证、检测记录、承载力、功能试验或精确距离。

每个风险输出一个紧贴可见违规证据的范围框。坐标基于整张输入图片，采用0到1000归一化整数，原点在左上角，格式为 [x1,y1,x2,y2]，其中x向右、y向下，x1/y1为左上角，x2/y2为右下角。范围框应包住实际违规对象/证据，不要框整张图或无关区域。若一条风险涉及彼此分离的多个位置，拆成多个风险项分别定位。无法从图中可靠定位时 bbox_2d_1000 必须为 null，禁止猜坐标；有可定位证据时必须给有效坐标。

法规要求：仅当能确认规范名称、编号和条款与风险确实对应时才填写 citation_key。只能从以下已核验条款目录选择；目录没有明确对应条文或无法确认时写 UNKNOWN。不得编造标准编号、条款原文或查询链接。程序会将合法键替换为目录中的完整、可查询法规文本。
{regulation_catalog_for_prompt(scene_key)}

状态：CLEAR表示违规状态有直接清晰证据，item和evidence中不得出现“疑似、可能、无法确认、无法判断”等不确定措辞；SUSPECTED表示存在可见线索但仍有不确定性；NOT_ASSESSABLE表示图片不能判断，此时item应描述“某状态无法判断”，不要把未确认事实写成确定违规名称。
只输出一个合法JSON对象，不输出Markdown、代码围栏或解释，最多5条风险。格式：
{{
  "scene_type": "{scene_key}",
  "scene": "场景中文名称",
  "issues": [
    {{
      "status": "CLEAR/SUSPECTED/NOT_ASSESSABLE",
      "item": "简明违规项",
      "target": "违规对象",
      "position": "图中方位",
      "evidence": "图片中直接可见的现象",
      "citation_key": "已核验条款键或UNKNOWN",
      "confidence": 0.0,
      "target_visibility": "complete/partial/occluded/not_visible",
      "evidence_level": "direct/partial/occluded/not_visible/ambiguous",
      "citation_confidence": "high/medium/low/unknown",
      "bbox_2d_1000": [100, 200, 400, 650]
    }}
  ]
}}
无风险时 issues 为空数组。"""

REVIEW_SYSTEM_PROMPT = """你是施工安全事实逻辑复核模型。

你只接收 Flash 已抽取的 JSON 字段，不要重新看图，也不要声称看到了图片。
只检查字段之间是否自相矛盾，不能新增图片事实。

检查规则：
1. 父对象不存在时，依赖该对象的子状态必须不适用。
2. 对象存在但关键部位被遮挡时，子状态应为 uncertain，不得写成明确违规。
3. 主箱门和插座/接口是独立字段；只检查 JSON 是否自相矛盾，不依据关键词替换视觉模型对其中一个部件的判断。
4. 吊装绳索不等于电气电缆。
5. 支腿、垫板、连墙件、附着、吊钩防脱、层门等未入镜，不等于缺失。
6. 静态图片不能验证功能试验、承载力、额定载荷、资质、台账或精确距离。
7. 只能把已有的 yes 降级为 uncertain 或 not_applicable，不能把 uncertain/no 改成 yes。

只输出需要人工复核的字段，不要重复输出完整 facts：
{
  "review_flags": {
    "字段名": "结构性冲突的简短原因"
  },
  "review_notes": ["最多1条，最多40字"]
}
如果没有需要复核，输出 {"review_flags": {}, "review_notes": []}。"""


GENERIC_FACT_EXTRACTION_SYSTEM_PROMPT = """你是施工现场安全图片快速检测模型。

你需要同时完成三件事：
1. 从图片中抽取当前场景中可直接支持的视觉事实；
2. 基于你的施工安全规范知识，对图片中清晰可见的风险进行安全检查判断。
3. 对每条风险进行证据自检，说明目标对象是否完整可见，以及规范引用把握程度。

必须只输出一个合法 JSON 对象，不要输出 Markdown、解释文字或思考过程。

高精度检查协议：
1. 先建立图片内对象清单，再判断风险。逐个检查人员、作业设备、危险区域、材料、通道和防护设施。
2. 对每个人分别判断：头部、躯干、手脚和作业位置是否完整入镜。头部被裁切、遮挡、背向过小或画面没有头部时，禁止写“未戴安全帽”；只能写“头部未完整入镜，无法判断安全帽状态”。
3. “没有看到”不等于“没有”。对安全帽、反光衣、安全带、防护栏、接地、灭火器、防脱装置等缺失类结论，必须先确认对应区域完整可见。
4. 任何空间关系都必须有几何证据：只有人员与吊物下方、临边、洞口、机械危险区或通道的相对位置清楚，才能判断“进入/位于/靠近”；仅凭画面边缘、重叠或距离感觉不能下结论。
5. 组合设备必须由视觉模型拆分识别。配电箱主箱门、插座盖、插头、接线端子和电缆接口分别判断，不能仅凭位置或关键词推断，也不能把一个部件的状态复制给另一个部件。
6. 对光影、反光、阴影、压缩伪影、颜色相似和遮挡保持怀疑；不能把它们当成积水、孔洞、开门、裸露、裂缝或变形的直接证据。
7. 只依据图片中清晰可见的静态证据。设计计算、额定载荷、承载力、功能试验、资质、作业票、检测记录和精确距离不能由单张图片证明。
8. 不要把吊装钢丝绳、吊带、软管或脚手架构件识别为电气电缆。
9. 在配电箱中，插座、插排、断路器、端子板、导线、线卡和接地排都是电气部件；只要能看出其属于电气系统，就不能因为颜色白、形状小或位于箱底而称为“塑料瓶、杂物”。只有与电气系统无关且形态清楚的物品才能判定为箱内杂物。
10. “未看到标签/接线图/锁具/防护套”不等于“缺少”。只有对应安装位置、箱门和箱体区域完整清晰可见时，才能判定缺失；箱门未入镜时禁止判定“箱门无锁”。
11. facts 字段是辅助证据，不是有限规则库；issues 才是你的安全检查判断结果。不要因为字段没有列出某类风险就忽略它。
12. 至少扫描以下风险类别：个人防护、人员站位和作业行为、临边洞口、高处作业、临时用电、机械设备、吊装、脚手架、基坑、消防、材料堆放、通道和警戒隔离。
13. 对每个 issue 做一次自检：对象是否真的存在、关键部位是否完整可见、空间关系是否清楚、现象是否由图片直接支持、规范是否与风险类型匹配。
14. 不要自由生成最终法规文本。模型只能从下面的已核验条款目录中选择 `citation_key`；无法直接对应时写 UNKNOWN。代码会根据合法键补全标准名称、条款号、条文原文和查询链接。
15. 同一个根本风险只输出一次，不要把同一现场现象造成的不同后果拆成多个重复问题。
16. `risk_key` 必须使用稳定、简短的根本风险名称，不要加入位置、后果或语气变化；两个模型看到同一根本风险时应尽量返回相同 `risk_key`。
17. 不要把某个场景字段当成检查范围上限；现场图片中出现的其他安全风险也要由你判断并写入 issues。

状态定义：
- CLEAR：对象和关键部位完整可见，违规状态有直接证据，才允许使用。
- SUSPECTED：有部分直接证据，但存在遮挡、角度、尺度或规范适用性不确定。
- NOT_ASSESSABLE：关键部位未入镜、被遮挡、过小、严重反光，或必须依赖测量/动态过程。
- `target_visibility` 必须写 complete/partial/occluded/not_visible；`evidence_level` 必须写 direct/partial/occluded/not_visible/ambiguous。

{scene_instructions}

JSON 格式：
{{
  "scene_type": "{scene_type}",
  "scene": "场景中文名称",
  "image_quality": {{
    "clarity": "clear/blurry/uncertain",
    "glare": true,
    "occlusion": true,
    "note": "最多40字"
  }},
  "facts": {{
{fact_schema}
  }},
  "issues": [
    {{
      "status": "CLEAR/SUSPECTED/NOT_ASSESSABLE",
      "item": "问题名称",
      "risk_key": "同一根本风险的简短名称，用于跨模型去重",
      "target": "风险对象或人员编号",
      "position": "图片中的大致位置，最多20字",
      "evidence": "图片中直接可见的事实，最多60字",
      "citation_key": "已核验条款键或UNKNOWN",
      "confidence": 0.0,
      "target_visibility": "complete/partial/occluded/not_visible",
      "evidence_level": "direct/partial/occluded/not_visible/ambiguous",
      "citation_confidence": "high/medium/low/unknown",
      "needs_review": true
    }}
  ]
}}

最多输出5个 issues；优先覆盖不同风险类别，同时保留证据清楚的疑似风险。"""


def build_fact_extraction_system_prompt(scene: str | None) -> str:
    scene_key = normalize_scene(scene)
    return GENERIC_FACT_EXTRACTION_SYSTEM_PROMPT.format(
        scene_instructions=scene_prompt_instructions(scene_key),
        scene_type=scene_key,
        fact_schema=scene_fact_schema(scene_key),
    )


def build_peer_vision_system_prompt(scene: str | None) -> str:
    scene_key = normalize_scene(scene)
    profile = get_scene_profile(scene_key)
    return f"""你是多模型平权投票中的独立施工安全视觉检测模型。场景：{profile['label']}（{scene_key}）。
目标：只用图片中清晰可见的静态证据，完成一次覆盖优先的安全检查。不要只挑最严重的少数项目；最多返回4个彼此独立、证据足够描述的风险候选，包含合理但需要复核的少数候选。
通用扫描范围：先盘点人员、设备、危险源、临边洞口、高处、临时用电、机械、吊装、脚手架、基坑、消防、材料、通道和警戒隔离，再判断风险。场景不是检查范围上限。
判定边界：未入镜、遮挡、反光、阴影、模糊、过小、需测量或动态验证的内容不得判为明确违规；头部不完整可见不得判未戴安全帽；“未看到”不等于“缺失”。
空间与对象原则：每条风险必须写明目标对象和可定位位置；只有相对位置、连接关系或部件状态在图中清楚，才能判断进入、靠近、外露、拖地或缺失。组合设备按实际部件分别识别，不根据关键词复制状态。
配电箱识别原则：主箱门、插座盖、插头、端子、电缆接口分别判断；插座、插排、断路器、端子板、导线和接地排属于电气部件，不得误报为塑料瓶或普通杂物。只有与电气系统无关且形态清楚的物品才可选择 TEMP_ELEC_BOX_CLEAN。
缺失类风险原则：只有对应安装区域完整清晰可见时，才能判断防护、锁具、挡脚板、接地、警示等缺失；关键区域不可见时写 NOT_ASSESSABLE。
输出原则：同一现场现象只写一次；不同对象或不同位置即使对应同一法规也要分别描述，不要因为法规编号相同而合并。risk_key 使用稳定的根本风险名称，不加入位置和后果。只输出JSON，不要解释。法规引用只能使用给出的 citation_key，禁止编造标准编号或条文。
状态：CLEAR=直接证据；SUSPECTED=部分证据；NOT_ASSESSABLE=图片无法判断。
{regulation_catalog_for_prompt(scene_key)}
{{
  "scene_type": "{scene_key}",
  "issues": [
    {{
      "status": "CLEAR/SUSPECTED/NOT_ASSESSABLE",
      "item": "问题名称",
      "risk_key": "根本风险短名称",
      "target": "风险对象，例如左侧人员/箱底电缆/洞口边缘",
      "position": "图片中的大致位置，最多20字",
      "evidence": "直接证据，最多35字",
      "citation_key": "已核验条款键或UNKNOWN",
      "confidence": 0.0,
      "target_visibility": "complete/partial/occluded/not_visible",
      "evidence_level": "direct/partial/ambiguous/occluded/not_visible"
    }}
  ]
}}
最多4个issues；无证据充分的风险时issues为空数组。"""


def build_qwen_vision_system_prompt(scene: str | None) -> str:
    return build_peer_vision_system_prompt(scene)


def build_third_vision_system_prompt(scene: str | None) -> str:
    return build_peer_vision_system_prompt(scene)


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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def effective_max_tokens(configured: int, reasoning_effort: str) -> int:
    """Reserve enough completion budget for reasoning and the final report."""
    minimums = {
        "none": 600,
        "low": 900,
        "high": 2400,
        "max": 3200,
    }
    return max(configured, minimums[reasoning_effort])


def remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.perf_counter())


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
        "thinking": {"type": "disabled" if reasoning_effort == "none" else "enabled"},
    }
    if reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort
    return payload


def build_qwen_review_payload(
    system_prompt: str,
    user_text: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Build an OpenAI-compatible Qwen request with thinking disabled.

    Qwen uses ``enable_thinking`` in ``extra_body`` for this switch. The
    reviewer only checks Flash JSON, so disabling thinking keeps this second
    request short and predictable.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "enable_thinking": False,
        "extra_body": {"enable_thinking": False},
    }


def build_qwen_vision_payload(
    image_data_url: str,
    system_prompt: str,
    user_text: str,
    model: str,
    image_detail: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    payload = {
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
    }
    # Qwen3 uses the top-level switch on the OpenAI-compatible endpoint.
    # Older Qwen vision models use extra_body instead.
    if model.strip().lower().startswith("qwen3"):
        payload["enable_thinking"] = False
    else:
        payload["extra_body"] = {"enable_thinking": False}
    return payload


def build_openai_vision_payload(
    image_data_url: str,
    system_prompt: str,
    user_text: str,
    model: str,
    image_detail: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Build a plain OpenAI-compatible vision payload for third-party models."""
    payload = {
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
    }
    model_name = model.strip().lower()
    if model_name.startswith("glm-5.3"):
        # GLM-5.3-Flash requires thinking to stay enabled, but low effort is
        # fast enough for this third-vote path and avoids empty final content.
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = "low"
        payload["clear_thinking"] = True
    elif model_name.startswith("kimi-"):
        # Raw HTTP requests must put Kimi's thinking switch at the top level;
        # ``extra_body`` is only merged automatically by some SDKs.
        payload["temperature"] = 0.6
        payload["thinking"] = {"type": "disabled"}
    elif model_name.startswith("doubao-") or model_name.startswith("ep-"):
        # Keep the Doubao peer on the fast path. Endpoint IDs also use this
        # branch because Ark commonly requires an ep-* model identifier.
        payload["thinking"] = {"type": "disabled"}
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


def strict_issue_quality_gate(issue: dict[str, Any]) -> dict[str, Any]:
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

    result["item"] = item[:80]
    result["risk_key"] = str(result.get("risk_key") or item)[:80]
    result["position"] = str(result.get("position", ""))[:40]
    result["evidence"] = evidence[:160]
    result["target"] = str(result.get("target", "未指定对象"))[:60]
    result["target_visibility"] = visibility
    result["evidence_level"] = evidence_level
    result["citation_confidence"] = normalize_issue_meta(
        result.get("citation_confidence"), "citation"
    )
    citation_key = str(result.get("citation_key") or "UNKNOWN").strip().upper()
    regulation = resolve_regulation(citation_key)
    result["citation_key"] = citation_key if regulation else "UNKNOWN"
    result["rule"] = format_regulation(result["citation_key"])
    if not regulation:
        result["citation_confidence"] = "unknown"
    result["status"] = status if status in {"CLEAR", "SUSPECTED", "NOT_ASSESSABLE"} else "SUSPECTED"
    result["needs_review"] = bool(
        result.get("needs_review", result["status"] != "CLEAR")
        or result["status"] != "CLEAR"
    )
    return result


def normalize_fact_result(
    data: dict[str, Any],
    scene_override: str | None = None,
    original_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    detected_scene = normalize_scene(
        scene_override or str(data.get("scene_type") or data.get("scene") or "")
    )
    facts = data.get("facts")
    if not isinstance(facts, dict):
        facts = {}
    profile = get_scene_profile(detected_scene)
    field_names = tuple(profile.get("fields", {}).keys())
    normalized_facts = {
        key: ynu(facts.get(key))
        for key in field_names
    }
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
                "target": str(issue.get("target", "未指定对象"))[:60],
                "position": str(issue.get("position", ""))[:40],
                "evidence": str(issue.get("evidence", "图片可见风险线索，需复核"))[:160],
                "citation_key": str(issue.get("citation_key", "UNKNOWN"))[:80],
                "confidence": max(0.0, min(1.0, confidence)),
                # Missing evidence metadata must fail closed. The model remains
                # the safety judge, but an omitted visibility/evidence field
                # cannot silently turn a claim into a clear violation.
                "target_visibility": issue.get("target_visibility", "partial"),
                "evidence_level": issue.get("evidence_level", "ambiguous"),
                "citation_confidence": issue.get("citation_confidence", "unknown"),
                "needs_review": bool(issue.get("needs_review", status != "CLEAR")),
            }
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
    return {
        "scene_type": detected_scene,
        "scene": str(data.get("scene", profile.get("label", "未识别"))),
        "image_quality": image_quality,
        "facts": normalized_facts,
        "issues": normalized_issues,
    }


def issue_status_zh(status: str) -> str:
    return {
        "CLEAR": "明确违规",
        "SUSPECTED": "疑似违规",
        "NOT_ASSESSABLE": "不可判断",
    }.get(str(status).upper(), "疑似违规")


def issue_key(issue: dict[str, Any]) -> str:
    # Let the model provide a stable root-risk label. The fallback only
    # normalizes the model's own title; it contains no domain keyword table.
    text = str(issue.get("risk_key") or issue.get("item") or "安全风险")
    text = re.sub(r"[\s，。；;:：、,.《》<>（）()【】\\[\\]-]+", "", text).lower()
    return text[:80]


def issue_text_ngrams(issue: dict[str, Any]) -> set[str]:
    """Build generic character n-grams for cross-model duplicate detection."""
    text = " ".join(
        str(issue.get(name, ""))
        for name in ("risk_key", "item", "target", "position", "evidence")
    )
    text = re.sub(r"[\s，。；;:：、,.《》<>（）()【】\\[\\]!?！？\"'“”‘’]+", "", text)
    return {text[index : index + 2] for index in range(max(0, len(text) - 1))}


def issue_target_ngrams(issue: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(issue.get(name, ""))
        for name in ("target", "position")
    )
    text = re.sub(r"[\s，。；;:：、,.《》<>（）()【】\\[\\]!?！？\"'“”‘’]+", "", text)
    return {text[index : index + 2] for index in range(max(0, len(text) - 1))}


def issue_key_is_generic(issue: dict[str, Any]) -> bool:
    key = issue_key(issue)
    return not key or key in {"安全风险", "风险", "违规", "疑似违规", "安全问题"}


def issue_targets_are_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_target = issue_target_ngrams(left)
    right_target = issue_target_ngrams(right)
    if not left_target or not right_target:
        return True
    if left_target == right_target:
        return True
    common = len(left_target & right_target)
    overlap = common / min(len(left_target), len(right_target))
    return common >= 2 and overlap >= 0.25


def issues_are_related(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Find likely duplicate wording without knowing any safety domain."""
    left_key = issue_key(left)
    right_key = issue_key(right)
    if (
        left_key
        and left_key == right_key
        and not issue_key_is_generic(left)
        and issue_targets_are_compatible(left, right)
    ):
        return True
    left_grams = issue_text_ngrams(left)
    right_grams = issue_text_ngrams(right)
    if not left_grams or not right_grams:
        return False
    common = len(left_grams & right_grams)
    overlap = common / min(len(left_grams), len(right_grams))
    # A regulation identifier alone is never a duplicate signal. Require
    # stronger wording overlap and compatible target/position context so that
    # two different objects under the same article remain separate candidates.
    return (
        common >= 6
        and overlap >= 0.28
        and issue_targets_are_compatible(left, right)
    )


def status_vote_score(status: str) -> float:
    return {
        "CLEAR": 1.0,
        "SUSPECTED": 0.65,
        "NOT_ASSESSABLE": 0.25,
    }.get(str(status).upper(), 0.5)


def issue_evidence_score(issue: dict[str, Any]) -> float:
    evidence = normalize_issue_meta(issue.get("evidence_level"), "evidence")
    visibility = normalize_issue_meta(issue.get("target_visibility"), "visibility")
    score = {
        "direct": 0.9,
        "partial": 0.65,
        "ambiguous": 0.4,
        "occluded": 0.25,
        "not_visible": 0.15,
    }.get(evidence, 0.45)
    score += {
        "complete": 0.05,
        "partial": 0.0,
        "occluded": -0.12,
        "not_visible": -0.18,
    }.get(visibility, 0.0)
    return max(0.0, min(1.0, score))


def issue_confidence_value(issue: dict[str, Any]) -> float:
    try:
        confidence = float(issue.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 0.0:
        confidence = {
            "CLEAR": 0.82,
            "SUSPECTED": 0.6,
            "NOT_ASSESSABLE": 0.35,
        }.get(str(issue.get("status", "SUSPECTED")).upper(), 0.5)
    return max(0.0, min(1.0, confidence))


def per_model_issue_score(issue: dict[str, Any]) -> float:
    status_score = status_vote_score(str(issue.get("status", "SUSPECTED")))
    confidence_score = issue_confidence_value(issue)
    evidence_score = issue_evidence_score(issue)
    return round(
        max(0.0, min(1.0, 0.45 * status_score + 0.35 * confidence_score + 0.20 * evidence_score)),
        3,
    )


def apply_vote_summary(
    issue: dict[str, Any],
    total_models: int,
    completed_model_count: int | None = None,
) -> dict[str, Any]:
    result = dict(issue)
    sources = list(dict.fromkeys(result.get("model_sources", [])))
    statuses = result.get("model_statuses", {})
    scores = result.get("model_scores", {})
    support_count = len(sources)
    total = max(1, total_models)
    completed = max(
        1,
        min(
            total,
            completed_model_count if completed_model_count is not None else total,
        ),
    )
    avg_score = (
        sum(float(scores.get(source, 0.0)) for source in sources) / support_count
        if support_count
        else 0.0
    )
    consensus_score = support_count / completed
    configured_support_rate = support_count / total
    # Keep single-model visual confidence separate from ensemble agreement.
    # A fast response from only part of the configured roster must not look
    # like a complete ensemble decision merely because those responses agree.
    raw_fused_confidence = (
        0.45 * avg_score
        + 0.35 * consensus_score
        + 0.20 * configured_support_rate
    )
    completed_rate = completed / total
    availability_factor = 0.35 + 0.65 * completed_rate
    fused_confidence = round(
        max(
            0.0,
            min(
                1.0,
                raw_fused_confidence * availability_factor,
            ),
        ),
        3,
    )
    majority_threshold = (total // 2) + 1
    majority_supported = support_count >= majority_threshold

    clear_votes = sum(1 for status in statuses.values() if str(status).upper() == "CLEAR")
    if majority_supported and clear_votes >= majority_threshold and fused_confidence >= 0.68:
        vote_status = "CLEAR"
    elif support_count >= 1 and (
        fused_confidence >= 0.35 or avg_score >= 0.55
    ):
        vote_status = "SUSPECTED"
    else:
        vote_status = "NOT_ASSESSABLE"

    # A single-model clear claim is useful, but not strong enough for final
    # "clear violation" output in the three-model voting scheme.
    if total >= 3 and support_count == 1 and vote_status == "CLEAR":
        vote_status = "SUSPECTED"

    result["model_sources"] = sources
    result["vote_support"] = support_count
    result["vote_total"] = total
    result["vote_completed"] = completed
    result["vote_majority"] = majority_supported
    result["model_evidence_confidence"] = round(avg_score, 3)
    result["consensus_score"] = round(consensus_score, 3)
    result["configured_support_rate"] = round(configured_support_rate, 3)
    result["model_availability_rate"] = round(completed_rate, 3)
    result["vote_confidence"] = fused_confidence
    result["vote_status"] = vote_status
    result["vote_label"] = (
        "多数模型支持"
        if majority_supported
        else "单模型发现，待人工复核"
        if support_count == 1
        else "少数模型发现，待人工复核"
    )
    result["needs_review"] = bool(result.get("needs_review", False) or not majority_supported)
    return result


def merge_model_issues(
    model_issue_sets: list[tuple[str, list[dict[str, Any]]]],
    total_models: int,
    model_roster: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge model-judged safety issues and compute vote confidence."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    roster = model_roster or [
        {"source": source, "model": source, "status": "completed"}
        for source, _ in model_issue_sets
    ]
    completed_model_count = sum(
        1 for model_info in roster
        if str(model_info.get("status", "completed")) == "completed"
    )

    for source, issues in model_issue_sets:
        for issue in issues:
            key = issue_key(issue)
            if key not in merged:
                for existing_key in order:
                    existing = merged[existing_key]
                    if source not in existing.get("model_sources", []) and issues_are_related(existing, issue):
                        key = existing_key
                        break
            if key not in merged:
                copied = dict(issue)
                copied["model_sources"] = [source]
                copied["model_statuses"] = {source: issue.get("status", "SUSPECTED")}
                copied["model_confidences"] = {source: issue_confidence_value(issue)}
                copied["model_scores"] = {source: per_model_issue_score(issue)}
                copied["model_agreement"] = "单模型"
                merged[key] = copied
                order.append(key)
                continue

            existing = merged[key]
            existing["confidence"] = max(issue_confidence_value(existing), issue_confidence_value(issue))
            existing["needs_review"] = bool(existing.get("needs_review", True) or issue.get("needs_review", True))
            existing.setdefault("model_sources", []).append(source)
            existing.setdefault("model_statuses", {})[source] = issue.get("status", "SUSPECTED")
            existing.setdefault("model_confidences", {})[source] = issue_confidence_value(issue)
            existing.setdefault("model_scores", {})[source] = per_model_issue_score(issue)
            existing["model_agreement"] = (
                "一致"
                if len(set(existing["model_statuses"].values())) == 1
                else "分歧"
            )

    status_rank = {"CLEAR": 0, "SUSPECTED": 1, "NOT_ASSESSABLE": 2}
    ordered = [merged[key] for key in order]
    for index, issue in enumerate(ordered):
        issue = strict_issue_quality_gate(issue)
        issue = apply_vote_summary(issue, total_models, completed_model_count)
        named_votes: list[dict[str, Any]] = []
        for model_info in roster:
            source = str(model_info.get("source", "unknown"))
            model = str(model_info.get("model", source))
            call_status = str(model_info.get("status", "completed"))
            display = model_display_name(source, model)
            if source in issue.get("model_sources", []):
                named_votes.append(
                    {
                        "source": source,
                        "model": model,
                        "display_name": display,
                        "vote": "支持",
                        "status": str(issue.get("model_statuses", {}).get(source, "SUSPECTED")),
                        "score": round(float(issue.get("model_scores", {}).get(source, 0.0)), 3),
                    }
                )
            elif call_status == "completed":
                named_votes.append(
                    {
                        "source": source,
                        "model": model,
                        "display_name": display,
                        "vote": "未支持",
                        "status": "未发现该风险",
                    }
                )
            else:
                named_votes.append(
                    {
                        "source": source,
                        "model": model,
                        "display_name": display,
                        "vote": "未返回",
                        "status": str(model_info.get("error") or "调用失败或超时")[:80],
                    }
                )
        issue["named_votes"] = named_votes
        ordered[index] = issue
    ordered.sort(
        key=lambda issue: (
            0 if issue.get("vote_majority") else 1,
            status_rank.get(str(issue.get("vote_status", issue.get("status", "SUSPECTED"))).upper(), 1),
            -float(issue.get("model_evidence_confidence", 0.0)),
            -float(issue.get("vote_confidence", 0.0)),
        )
    )
    majority = [issue for issue in ordered if issue.get("vote_majority")]
    minority = [issue for issue in ordered if not issue.get("vote_majority")]
    # Keep a visible review lane for minority candidates instead of allowing
    # majority sorting to silently erase plausible single-model findings.
    if majority:
        return majority[:5] + minority[:3]
    return ordered[:8]


def needs_strong_review(facts_data: dict[str, Any], local_quality: dict[str, Any]) -> bool:
    """Decide whether a text-only consistency pass is worth its latency.

    This reviewer cannot see the image. It must never be used to make a
    domain-specific visual decision, so only model-reported structure and
    evidence metadata can trigger it.
    """
    facts = facts_data.get("facts", {})
    issues = facts_data.get("issues", [])
    if not isinstance(facts, dict) or not isinstance(issues, list):
        return False

    # A low-quality image makes a metadata consistency pass useful, but it
    # still does not authorize the reviewer to invent or replace visual facts.
    poor_image = (
        local_quality.get("clarity") != "clear"
        or bool(local_quality.get("glare"))
        or bool(local_quality.get("dark"))
    )
    has_multiple_uncertain_facts = (
        sum(value == "uncertain" for value in facts.values()) >= 2
    )
    has_clear_issue_without_direct_evidence = any(
        str(issue.get("status", "")).upper() == "CLEAR"
        and normalize_issue_meta(issue.get("evidence_level"), "evidence") != "direct"
        for issue in issues
        if isinstance(issue, dict)
    )
    has_assessability_conflict = any(
        str(issue.get("status", "")).upper() == "NOT_ASSESSABLE"
        and normalize_issue_meta(issue.get("evidence_level"), "evidence") == "direct"
        for issue in issues
        if isinstance(issue, dict)
    )
    return poor_image or has_multiple_uncertain_facts or has_clear_issue_without_direct_evidence or has_assessability_conflict


def apply_scene_gate(
    facts_data: dict[str, Any],
    scene: str | None,
    prompt: str,
) -> dict[str, Any]:
    """Attach the canonical scene without disabling other safety domains."""
    requested_scene = normalize_scene(scene)
    detected_scene = normalize_scene(
        str(facts_data.get("scene_type") or facts_data.get("scene") or "")
    )
    selected_scene = requested_scene if requested_scene != "AUTO" else detected_scene
    result = dict(facts_data)
    result["scene_type"] = selected_scene
    result["scene_profile"] = get_scene_profile(selected_scene).get("label", selected_scene)
    return result


def merge_review(primary: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Attach text-only review findings without changing visual facts.

    The reviewer does not receive the image. Even a conservative downgrade
    would still be a second model deciding a visual fact from JSON alone, so
    the primary model's facts and issues remain authoritative. Review output is
    retained as an auditable note for the user.
    """
    review_flags = review.get("review_flags")
    if not isinstance(review_flags, dict):
        # Accept the old key while migrating existing deployments, but never
        # apply it as a visual fact rewrite.
        review_flags = review.get("downgrades")
    if not isinstance(review_flags, dict):
        review_flags = {}
    merged = dict(primary)
    if review_flags:
        merged["review_flags"] = {
            str(key)[:80]: str(value)[:30]
            for key, value in list(review_flags.items())[:20]
        }
    notes = review.get("review_notes")
    if isinstance(notes, list):
        merged["review_notes"] = [str(note)[:80] for note in notes[:3]]
    return merged


def merge_multi_vision_results(
    primary: dict[str, Any],
    secondary_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Keep primary facts and fuse model issues through voting."""
    scene_key = normalize_scene(
        primary.get("scene_type")
        or next((result.get("scene_type") for _, result in secondary_results if result), None)
    )
    profile = get_scene_profile(scene_key)
    field_names = set(profile.get("fields", {}).keys())
    primary_facts = primary.get("facts", {}) if isinstance(primary.get("facts"), dict) else {}
    model_facts: dict[str, dict[str, str]] = {
        "deepseek": {field: ynu(primary_facts.get(field)) for field in field_names}
    }
    fact_comparison_sources = {"deepseek"}
    for source, result in secondary_results:
        facts = result.get("facts", {}) if isinstance(result.get("facts"), dict) else {}
        model_facts[source] = {field: ynu(facts.get(field)) for field in field_names}
        if any(value in {"yes", "no", "uncertain"} for value in model_facts[source].values()):
            fact_comparison_sources.add(source)

    disagreements: dict[str, dict[str, str]] = {}
    for field in field_names:
        values = {
            source: facts.get(field, "not_applicable")
            for source, facts in model_facts.items()
            if source in fact_comparison_sources
        }
        if len(values) > 1 and len(set(values.values())) > 1:
            disagreements[field] = values

    model_issue_sets: list[tuple[str, list[dict[str, Any]]]] = [
        ("deepseek", primary.get("issues", []) if isinstance(primary.get("issues"), list) else [])
    ]
    for source, result in secondary_results:
        model_issue_sets.append(
            (source, result.get("issues", []) if isinstance(result.get("issues"), list) else [])
        )

    merged = dict(primary)
    merged["facts"] = dict(primary_facts)
    merged["model_facts"] = model_facts
    merged["secondary_facts"] = {
        source: result.get("facts", {}) if isinstance(result.get("facts"), dict) else {}
        for source, result in secondary_results
    }
    total_models = 1 + len(secondary_results)
    merged["issues"] = merge_model_issues(model_issue_sets, total_models=total_models)
    merged["scene_type"] = scene_key
    merged["vision_review"] = {
        "secondary_models": [source for source, _ in secondary_results],
        "fact_disagreements": disagreements,
        "completed_models": list(model_facts),
        "total_models": total_models,
    }
    normalized = normalize_fact_result(merged, scene_key)
    normalized["issues"] = merged["issues"]
    normalized["secondary_facts"] = merged["secondary_facts"]
    normalized["model_facts"] = merged["model_facts"]
    normalized["vision_review"] = merged["vision_review"]
    return normalized


def merge_peer_vision_results(
    peer_results: list[tuple[str, dict[str, Any]]],
    scene: str | None = None,
    model_roster: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fuse equal-peer visual safety judgments through confidence voting."""
    scene_key = normalize_scene(
        scene
        or next((result.get("scene_type") for _, result in peer_results if result), None)
    )
    profile = get_scene_profile(scene_key)
    model_issue_sets = [
        (source, result.get("issues", []) if isinstance(result.get("issues"), list) else [])
        for source, result in peer_results
    ]
    total_models = max(1, len(model_roster or peer_results))
    merged_issues = merge_model_issues(
        model_issue_sets,
        total_models=total_models,
        model_roster=model_roster,
    )
    merged = {
        "scene_type": scene_key,
        "scene": profile.get("label", "未识别"),
        "facts": {},
        "issues": merged_issues,
        "vision_review": {
            "mode": f"{total_models}模型平权并行投票",
            "completed_models": [source for source, _ in peer_results],
            "total_models": total_models,
            "model_roster": model_roster or [
                {"source": source, "model": source, "status": "completed"}
                for source, _ in peer_results
            ],
            "fact_disagreements": {},
        },
    }
    normalized = normalize_fact_result(merged, scene_key)
    normalized["issues"] = merged_issues
    normalized["vision_review"] = merged["vision_review"]
    return normalized


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
        "报告格式版本：1.0",
        "检测模型：DeepSeek Flash（单模型单次视觉检测）",
        f"检查场景：{profile['label']}",
        f"主要参考规范：{profile['standards']}",
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
        status_code = str(issue.get("vote_status", issue.get("status", "SUSPECTED"))).upper()
        status = issue_status_zh(status_code)
        lines.append(f"{idx}. {status}：{issue.get('item', '安全风险线索')}")
        position = str(issue.get("position") or issue.get("target") or "").strip()
        if position:
            lines.append(f"• 目标位置：{position}")
        lines.append(f"• 现象描述：{issue.get('evidence', '图片可见风险线索，需复核')}")
        regulation_label = (
            "违反的具体安全条例"
            if status_code == "CLEAR"
            else "可能涉及的安全条例（尚未确认违反）"
        )
        lines.append(f"• {regulation_label}：{issue.get('rule', format_regulation('UNKNOWN'))}")
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

    review_notes = facts_data.get("review_notes")
    if isinstance(review_notes, list) and review_notes:
        notes.extend(str(note) for note in review_notes[:3])
    vision_review = facts_data.get("vision_review")
    if isinstance(vision_review, dict):
        notes.append("本报告由单个视觉大模型生成，没有进行多模型投票或交叉复核。")
    review_flags = facts_data.get("review_flags")
    if isinstance(review_flags, dict) and review_flags:
        notes.append("文本一致性检查发现结构性不一致，请人工复核相关字段。")

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


def run_vision_job(job: VisionJob) -> VisionCallResult:
    started = time.perf_counter()
    try:
        raw = call_deepseek(
            job.payload,
            api_key=job.api_key,
            base_url=job.base_url,
            timeout_seconds=job.timeout,
        )
        return VisionCallResult(
            source=job.source,
            model=job.model,
            raw=raw,
            elapsed=time.perf_counter() - started,
        )
    except Exception as exc:
        return VisionCallResult(
            source=job.source,
            model=job.model,
            raw=None,
            elapsed=time.perf_counter() - started,
            error=str(exc),
        )


def timing_lines(timings: dict[str, Any]) -> list[str]:
    lines = ["", "时间分配："]
    if "preprocess" in timings:
        lines.append(f"- 图片预处理与请求构造：{timings['preprocess']:.3f} 秒")
    model_calls = timings.get("model_calls")
    if isinstance(model_calls, list):
        for call in model_calls:
            status = "完成" if not call.get("error") else "失败"
            lines.append(
                f"- {call.get('source')} / {call.get('model')} 图片安全检测："
                f"{float(call.get('elapsed', 0.0)):.3f} 秒（{status}）"
            )
    if "parallel_models" in timings:
        model_count = len(timings.get("model_calls", []))
        lines.append(f"- {model_count}模型并行等待总时长：{timings['parallel_models']:.3f} 秒")
    if "parse_normalize" in timings:
        lines.append(f"- JSON 解析与证据规范化：{timings['parse_normalize']:.3f} 秒")
    if "vote_fusion" in timings:
        lines.append(f"- 置信度打分与投票融合：{timings['vote_fusion']:.3f} 秒")
    if "report_format" in timings:
        lines.append(f"- 报告格式化：{timings['report_format']:.3f} 秒")
    if "total" in timings:
        lines.append(f"- 总耗时：{timings['total']:.3f} 秒")
    return lines


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


def legacy_multi_model_main() -> int:
    """Inactive historical entrypoint retained for compatibility tests only."""
    # Always load the configuration beside this script, independent of the shell cwd.
    load_dotenv(Path(__file__).with_name(".env"))
    args = parse_args()

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
    review_model = os.getenv("QWEN_REVIEW_MODEL", "qwen3.8-flash").strip()
    qwen_vision_model = os.getenv("QWEN_VISION_MODEL", review_model).strip()
    qwen_base_url = os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    ).strip()
    third_vision_model = os.getenv("THIRD_VISION_MODEL", "glm-5.3-flash").strip()
    third_vision_base_url = os.getenv(
        "THIRD_VISION_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    ).strip()
    kimi_model = os.getenv("KIMI_VISION_MODEL", "kimi-k2.6").strip()
    kimi_base_url = os.getenv(
        "KIMI_BASE_URL",
        "https://api.moonshot.cn/v1/chat/completions",
    ).strip()
    doubao_model = os.getenv("DOUBAO_VISION_MODEL", "").strip()
    doubao_base_url = os.getenv(
        "DOUBAO_BASE_URL",
        "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    ).strip()
    enable_review = env_bool("ENABLE_STRONG_REVIEW", False)
    enable_qwen_vision = env_bool("ENABLE_QWEN_VISION_REVIEW", True)
    enable_third_vision = env_bool("ENABLE_THIRD_VISION_REVIEW", True)
    enable_kimi_vision = env_bool("ENABLE_KIMI_VISION", True)
    enable_doubao_vision = env_bool("ENABLE_DOUBAO_VISION", True)
    timeout_seconds = env_float("REQUEST_TIMEOUT_SECONDS", 8.0)
    deepseek_vision_timeout = env_float(
        "DEEPSEEK_VISION_TIMEOUT_SECONDS",
        env_float("PRIMARY_TIMEOUT_SECONDS", 4.2),
    )
    review_timeout = env_float("REVIEW_TIMEOUT_SECONDS", 3.2)
    qwen_vision_timeout = env_float("QWEN_VISION_TIMEOUT_SECONDS", 6.8)
    third_vision_timeout = env_float("THIRD_VISION_TIMEOUT_SECONDS", 7.2)
    kimi_vision_timeout = env_float("KIMI_VISION_TIMEOUT_SECONDS", 7.2)
    doubao_vision_timeout = env_float("DOUBAO_VISION_TIMEOUT_SECONDS", 7.2)
    image_detail = os.getenv("IMAGE_DETAIL", "low").strip()
    max_edge = env_int("MAX_IMAGE_EDGE", 768)
    jpeg_quality = env_int("JPEG_QUALITY", 70)
    qwen_max_edge = env_int("QWEN_MAX_IMAGE_EDGE", 512)
    qwen_jpeg_quality = env_int("QWEN_JPEG_QUALITY", 55)
    third_max_edge = env_int("THIRD_MAX_IMAGE_EDGE", 448)
    third_jpeg_quality = env_int("THIRD_JPEG_QUALITY", 50)
    kimi_max_edge = env_int("KIMI_MAX_IMAGE_EDGE", 512)
    kimi_jpeg_quality = env_int("KIMI_JPEG_QUALITY", 55)
    doubao_max_edge = env_int("DOUBAO_MAX_IMAGE_EDGE", 512)
    doubao_jpeg_quality = env_int("DOUBAO_JPEG_QUALITY", 55)
    max_tokens = env_int("MAX_TOKENS", 900)
    review_max_tokens = env_int("REVIEW_MAX_TOKENS", 350)
    qwen_vision_max_tokens = env_int("QWEN_VISION_MAX_TOKENS", 600)
    third_vision_max_tokens = env_int("THIRD_VISION_MAX_TOKENS", 400)
    kimi_vision_max_tokens = env_int("KIMI_VISION_MAX_TOKENS", 420)
    doubao_vision_max_tokens = env_int("DOUBAO_VISION_MAX_TOKENS", 420)
    temperature = env_float("TEMPERATURE", 0.0)
    reasoning_effort = os.getenv("REASONING_EFFORT", "none").strip().lower()
    if reasoning_effort not in {"none", "low", "high", "max"}:
        print("ERROR: REASONING_EFFORT must be none, low, high, or max.", file=sys.stderr)
        return 2
    request_max_tokens = effective_max_tokens(max_tokens, reasoning_effort)
    deepseek_output_tokens = min(request_max_tokens, 700) if reasoning_effort == "none" else request_max_tokens
    qwen_output_tokens = min(qwen_vision_max_tokens, 420)
    third_output_tokens = min(third_vision_max_tokens, 420)
    kimi_output_tokens = min(kimi_vision_max_tokens, 420)
    doubao_output_tokens = min(doubao_vision_max_tokens, 420)

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    requested_scene = normalize_scene(args.scene)

    try:
        started = time.perf_counter()
        deadline = started + timeout_seconds
        timings: dict[str, Any] = {}

        preprocess_started = time.perf_counter()
        local_quality = analyze_image_quality(image_path)
        peer_prompt = build_peer_vision_system_prompt(requested_scene)
        quality_hint = local_quality.get("note", "图片质量基本可用")
        peer_user_text = (
            f"已知检查场景：{args.scene.strip() or '未指定'}\n"
            f"用户任务：{args.prompt}\n"
            f"图像质量提示：{quality_hint}\n"
            "请快速完成图片安全检测，只返回JSON。"
        )
        image_data_url = prepare_image_data_url(image_path, max_edge=max_edge, quality=jpeg_quality)
        qwen_image_data_url = prepare_image_data_url(
            image_path,
            max_edge=qwen_max_edge,
            quality=qwen_jpeg_quality,
        )
        third_image_data_url = prepare_image_data_url(
            image_path,
            max_edge=third_max_edge,
            quality=third_jpeg_quality,
        )
        kimi_image_data_url = prepare_image_data_url(
            image_path,
            max_edge=kimi_max_edge,
            quality=kimi_jpeg_quality,
        )
        doubao_image_data_url = prepare_image_data_url(
            image_path,
            max_edge=doubao_max_edge,
            quality=doubao_jpeg_quality,
        )
        payload = build_vision_payload(
            image_data_url=image_data_url,
            system_prompt=peer_prompt,
            user_text=peer_user_text,
            model=model,
            image_detail=image_detail,
            max_tokens=deepseek_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        qwen_vision_payload = build_qwen_vision_payload(
            image_data_url=qwen_image_data_url,
            system_prompt=peer_prompt,
            user_text=peer_user_text,
            model=qwen_vision_model,
            image_detail=image_detail,
            max_tokens=qwen_output_tokens,
            temperature=temperature,
        )
        third_vision_payload = build_openai_vision_payload(
            image_data_url=third_image_data_url,
            system_prompt=peer_prompt,
            user_text=peer_user_text,
            model=third_vision_model,
            image_detail=image_detail,
            max_tokens=third_output_tokens,
            temperature=temperature,
        )
        kimi_vision_payload = build_openai_vision_payload(
            image_data_url=kimi_image_data_url,
            system_prompt=peer_prompt,
            user_text=peer_user_text,
            model=kimi_model,
            image_detail=image_detail,
            max_tokens=kimi_output_tokens,
            temperature=temperature,
        )
        doubao_vision_payload = build_openai_vision_payload(
            image_data_url=doubao_image_data_url,
            system_prompt=peer_prompt,
            user_text=peer_user_text,
            model=doubao_model,
            image_detail=image_detail,
            max_tokens=doubao_output_tokens,
            temperature=temperature,
        )
        timings["preprocess"] = time.perf_counter() - preprocess_started

        if args.dry_run:
            elapsed = time.perf_counter() - started
            approx_kb = len(image_data_url.encode("utf-8")) / 1024
            print("DRY RUN OK")
            print(f"image={image_path}")
            print(f"deepseek_model={model}")
            print(f"qwen_vision_model={qwen_vision_model}")
            print(f"third_vision_model={third_vision_model}")
            print(f"kimi_vision_model={kimi_model}")
            print(f"doubao_vision_model={doubao_model or '(not configured)'}")
            print(f"enable_qwen_vision={enable_qwen_vision}")
            print(f"enable_third_vision={enable_third_vision}")
            print(f"enable_kimi_vision={enable_kimi_vision}")
            print(f"enable_doubao_vision={enable_doubao_vision}")
            print(f"enable_text_review={enable_review}")
            print(f"timeout={timeout_seconds:.1f}s")
            print(f"deepseek_vision_timeout={deepseek_vision_timeout:.1f}s")
            print(f"review_timeout={review_timeout:.1f}s")
            print(f"qwen_vision_timeout={qwen_vision_timeout:.1f}s")
            print(f"third_vision_timeout={third_vision_timeout:.1f}s")
            print(f"kimi_vision_timeout={kimi_vision_timeout:.1f}s")
            print(f"doubao_vision_timeout={doubao_vision_timeout:.1f}s")
            print(f"image_detail={image_detail}")
            print(f"max_image_edge={max_edge}")
            print(f"jpeg_quality={jpeg_quality}")
            print(f"qwen_max_image_edge={qwen_max_edge}")
            print(f"qwen_jpeg_quality={qwen_jpeg_quality}")
            print(f"third_max_image_edge={third_max_edge}")
            print(f"third_jpeg_quality={third_jpeg_quality}")
            print(f"kimi_max_image_edge={kimi_max_edge}")
            print(f"kimi_jpeg_quality={kimi_jpeg_quality}")
            print(f"doubao_max_image_edge={doubao_max_edge}")
            print(f"doubao_jpeg_quality={doubao_jpeg_quality}")
            print(f"reasoning_effort={reasoning_effort}")
            print(f"configured_max_tokens={max_tokens}")
            print(f"deepseek_output_tokens={deepseek_output_tokens}")
            print(f"review_max_tokens={review_max_tokens}")
            print(f"qwen_output_tokens={qwen_output_tokens}")
            print(f"third_output_tokens={third_output_tokens}")
            print(f"kimi_output_tokens={kimi_output_tokens}")
            print(f"doubao_output_tokens={doubao_output_tokens}")
            print(f"scene_type={requested_scene}")
            print(f"local_quality={json.dumps(local_quality, ensure_ascii=False)}")
            print(f"deepseek_payload_image_data_url_size≈{approx_kb:.1f}KB")
            print(f"qwen_payload_image_data_url_size≈{len(qwen_image_data_url.encode('utf-8')) / 1024:.1f}KB")
            print(f"third_payload_image_data_url_size≈{len(third_image_data_url.encode('utf-8')) / 1024:.1f}KB")
            print(f"kimi_payload_image_data_url_size≈{len(kimi_image_data_url.encode('utf-8')) / 1024:.1f}KB")
            print(f"doubao_payload_image_data_url_size≈{len(doubao_image_data_url.encode('utf-8')) / 1024:.1f}KB")
            print(f"prepare_elapsed={elapsed:.2f}s")
            return 0

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key.startswith("sk-你的"):
            print("ERROR: Please set DEEPSEEK_API_KEY in .env first.", file=sys.stderr)
            return 2
        qwen_api_key = os.getenv("QWEN_API_KEY", "").strip()
        qwen_ready = bool(qwen_api_key and not qwen_api_key.startswith("sk-你的"))
        third_api_key = os.getenv("THIRD_VISION_API_KEY", "").strip()
        third_ready = bool(third_api_key and not third_api_key.startswith("sk-你的"))
        kimi_api_key = os.getenv("KIMI_API_KEY", "").strip()
        kimi_ready = bool(kimi_api_key and not kimi_api_key.startswith("sk-你的"))
        doubao_api_key = os.getenv("DOUBAO_API_KEY", "").strip()
        doubao_ready = bool(
            doubao_api_key
            and not doubao_api_key.startswith("sk-你的")
            and doubao_model
            and not doubao_model.startswith("ep-你的")
        )

        notes: list[str] = []
        # Reserve time after the slowest parallel call for JSON parsing,
        # fusion, report formatting, and saving the result.
        post_call_reserve = 0.45
        shared_call_budget = max(0.8, remaining_seconds(deadline) - post_call_reserve)
        deepseek_call_timeout = min(deepseek_vision_timeout, shared_call_budget)
        qwen_vision_call_timeout = min(qwen_vision_timeout, shared_call_budget)
        third_vision_call_timeout = min(third_vision_timeout, shared_call_budget)
        kimi_vision_call_timeout = min(kimi_vision_timeout, shared_call_budget)
        doubao_vision_call_timeout = min(doubao_vision_timeout, shared_call_budget)
        vision_jobs: list[VisionJob] = [
            VisionJob(
                source="deepseek",
                model=model,
                payload=payload,
                api_key=api_key,
                base_url=base_url,
                timeout=deepseek_call_timeout,
            )
        ]
        if enable_qwen_vision and qwen_ready:
            vision_jobs.append(
                VisionJob(
                    source="qwen",
                    model=qwen_vision_model,
                    payload=qwen_vision_payload,
                    api_key=qwen_api_key,
                    base_url=qwen_base_url,
                    timeout=qwen_vision_call_timeout,
                )
            )
        elif enable_qwen_vision:
            notes.append("未配置 QWEN_API_KEY，Qwen 未参与多模型投票。")
        if enable_third_vision and third_ready:
            vision_jobs.append(
                VisionJob(
                    source="glm",
                    model=third_vision_model,
                    payload=third_vision_payload,
                    api_key=third_api_key,
                    base_url=third_vision_base_url,
                    timeout=third_vision_call_timeout,
                )
            )
        elif enable_third_vision:
            notes.append("未配置 THIRD_VISION_API_KEY，GLM 未参与多模型投票。")
        if enable_kimi_vision and kimi_ready:
            vision_jobs.append(
                VisionJob(
                    source="kimi",
                    model=kimi_model,
                    payload=kimi_vision_payload,
                    api_key=kimi_api_key,
                    base_url=kimi_base_url,
                    timeout=kimi_vision_call_timeout,
                )
            )
        elif enable_kimi_vision:
            notes.append("未配置 KIMI_API_KEY，Kimi 未参与五模型投票。")
        if enable_doubao_vision and doubao_ready:
            vision_jobs.append(
                VisionJob(
                    source="doubao",
                    model=doubao_model,
                    payload=doubao_vision_payload,
                    api_key=doubao_api_key,
                    base_url=doubao_base_url,
                    timeout=doubao_vision_call_timeout,
                )
            )
        elif enable_doubao_vision:
            notes.append("未配置 DOUBAO_API_KEY 或 DOUBAO_VISION_MODEL，豆包未参与五模型投票。")

        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(vision_jobs)) as executor:
            futures = [executor.submit(run_vision_job, job) for job in vision_jobs]
            call_results = [future.result() for future in futures]
        expected_models = len(vision_jobs)
        model_roster = [
            {
                "source": call.source,
                "model": call.model,
                "status": "completed" if not call.error and call.raw else "failed",
                "error": call.error,
            }
            for call in call_results
        ]
        timings["parallel_models"] = time.perf_counter() - parallel_started
        timings["model_calls"] = [
            {
                "source": call.source,
                "model": call.model,
                "elapsed": call.elapsed,
                "error": call.error,
            }
            for call in call_results
        ]

        parse_started = time.perf_counter()
        peer_results: list[tuple[str, dict[str, Any]]] = []
        for call in call_results:
            if call.error or not call.raw:
                notes.append(f"{call.source} / {call.model} 未完成图片检测：{call.error}")
                continue
            try:
                parsed = normalize_fact_result(
                    extract_json_object(call.raw),
                    requested_scene if requested_scene != "AUTO" else None,
                )
                parsed = apply_scene_gate(
                    parsed,
                    scene=args.scene.strip() or None,
                    prompt=args.prompt,
                )
                peer_results.append((call.source, parsed))
            except Exception as exc:
                notes.append(f"{call.source} / {call.model} 返回格式异常，未参与投票：{exc}")
                for model_info in model_roster:
                    if model_info["source"] == call.source:
                        model_info["status"] = "failed"
                        model_info["error"] = f"JSON格式异常：{exc}"
                        break
        timings["parse_normalize"] = time.perf_counter() - parse_started

        if not peer_results:
            raise RuntimeError("所有视觉模型均未返回可解析的安全检测 JSON。")
        if len(peer_results) < expected_models:
            notes.append(
                f"本次有 {len(peer_results)}/{expected_models} 个视觉模型完成检测，"
                "未达到当前配置的完整投票。"
            )

        fusion_started = time.perf_counter()
        fact_result = merge_peer_vision_results(
            peer_results,
            scene=args.scene.strip() or None,
            model_roster=model_roster,
        )
        if notes:
            fact_result["review_notes"] = [str(note)[:120] for note in notes[:5]]
        timings["vote_fusion"] = time.perf_counter() - fusion_started

        report_started = time.perf_counter()
        elapsed = time.perf_counter() - started
        result = model_judged_report(
            fact_result,
            local_quality,
            elapsed=elapsed,
            scene=args.scene.strip() or None,
            prompt=args.prompt,
        )
        timings["report_format"] = time.perf_counter() - report_started
        timings["total"] = time.perf_counter() - started
        elapsed = timings["total"]
        if args.show_timing:
            result = f"{result}\n{chr(10).join(timing_lines(timings))}"
    except httpx.TimeoutException:
        print(f"ERROR: API call exceeded {timeout_seconds:.1f} seconds. Try smaller MAX_IMAGE_EDGE or MAX_TOKENS.", file=sys.stderr)
        return 1
    except StopIteration as exc:
        result = str(exc)
        elapsed = time.perf_counter() - started
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if elapsed > timeout_seconds:
        print(f"\nWARNING: Total elapsed time exceeded {timeout_seconds:.1f} seconds.", file=sys.stderr)

    if not args.no_save and not args.json_output:
        saved_path = save_result(output_dir, image_path=image_path, content=result, elapsed=elapsed)
        print(f"\n结果已保存：{saved_path}")

    if args.json_output:
        response = {
            "status": "success",
            "scene": fact_result.get("scene"),
            "scene_type": fact_result.get("scene_type"),
            "elapsed_seconds": round(elapsed, 3),
            "issues": fact_result.get("issues", []),
            "vision_review": fact_result.get("vision_review", {}),
            "warnings": fact_result.get("review_notes", []),
            "timings": timings,
            "report": result,
        }
        print(json.dumps(response, ensure_ascii=False))
    else:
        print(f"\n耗时：{elapsed:.2f} 秒\n")
        print(result)
    return 0


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
    max_tokens = env_int("SINGLE_MODEL_MAX_TOKENS", env_int("MAX_TOKENS", 1100))
    temperature = env_float("TEMPERATURE", 0.0)
    reasoning_effort = os.getenv("REASONING_EFFORT", "none").strip().lower()
    if reasoning_effort not in {"none", "low", "high", "max"}:
        print("ERROR: REASONING_EFFORT must be none, low, high, or max.", file=sys.stderr)
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
                "format_version": "1.0",
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
            print("ERROR: Please set DEEPSEEK_API_KEY in .env first.", file=sys.stderr)
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
        detected = apply_scene_gate(detected, scene=scene_override, prompt=args.prompt)
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
            scene=scene_override or detected.get("scene_type"),
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
            report += f"- JSON解析与坐标换算：{timings['parse_normalize']:.3f} 秒\n"
            report += f"- 报告格式化：{timings['report_format']:.3f} 秒\n"
            report += f"- 总耗时：{timings['total']:.3f} 秒"
    except httpx.TimeoutException:
        print(f"ERROR: DeepSeek Flash request exceeded the time budget ({timeout_seconds:.1f}s total).", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    elapsed = timings["total"]
    if elapsed > timeout_seconds:
        print(f"WARNING: Total elapsed time exceeded {timeout_seconds:.1f} seconds.", file=sys.stderr)

    response = {
        "status": "success",
        "format_version": "1.0",
        "output_format": "structured_json",
        "model": model,
        "scene": detected.get("scene"),
        "scene_type": detected.get("scene_type"),
        "image": {
            "width": image_size[0],
            "height": image_size[1],
            "coordinate_origin": "top_left",
            "coordinate_unit": "pixel",
        },
        "elapsed_seconds": elapsed,
        "issue_count": len(detected.get("issues", [])),
        "issues": detected.get("issues", []),
        "warnings": [],
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
