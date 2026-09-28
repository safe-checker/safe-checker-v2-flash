"""Verified construction-safety clause catalog used for report citations.

The vision models still decide what is visible and whether it is risky.
This module only supplies a queryable standard clause after a model selects
one of the explicit citation keys. Unknown keys deliberately fail closed.
"""

from __future__ import annotations

from typing import Any


JGJ_T_46_2024_SOURCE = "https://gf.cabr-fire.com/article-68828.htm"
JGJ_T_46_2024_SETTING_SOURCE = "https://gf.cabr-fire.com/m/article-68840.htm"
JGJ_T_46_2024_USE_SOURCE = "https://gf.cabr-fire.com/m/article-68842.htm"
JGJ_T_46_2024_CABLE_SOURCE = "https://gf.cabr-fire.com/m/article-68848.htm"
JGJ_T_46_2024_TITLE = "《建筑与市政工程施工现场临时用电安全技术标准》JGJ/T 46-2024"


VERIFIED_REGULATIONS: dict[str, dict[str, Any]] = {
    "TEMP_ELEC_CABLE_LAYOUT": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第6.2.3条",
        "text": "电缆线路应采用埋地或架空敷设，严禁沿地面明设，并应避免机械损伤和介质腐蚀。",
        "source": JGJ_T_46_2024_CABLE_SOURCE,
        "scope": "电缆沿地面明设、拖地或明显未采取架空/埋地保护",
    },
    "TEMP_ELEC_BOX_SPACE": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.5条",
        "text": "配电箱、开关箱周围应有足够2人同时工作的空间和通道，不得堆放任何妨碍操作和维修的物品，不得有灌木和杂草。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "配电箱周边操作空间或通道被明显占用",
    },
    "TEMP_ELEC_BOX_FIXED": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.7条",
        "text": "配电箱、开关箱应装设端正、牢固。固定式配电箱、开关箱的中心点与地面的垂直距离应为1.4m～1.6m。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "箱体明显倾倒、悬空不稳或固定失效",
    },
    "TEMP_ELEC_BOX_COMPONENT_FIX": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.9条",
        "text": "配电箱、开关箱内的电器（含插座）应按其规定位置固定在电器安装板上，且不得歪斜和松动。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "箱内电器安装板或安装方式有清晰可见的不符合要求情形",
    },
    "TEMP_ELEC_N_PE_TERMINALS": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.10条",
        "text": "配电箱、开关箱的电器安装板上必须分设N端子板和PE端子板；N端子板应与金属电器安装板绝缘，PE端子板应与金属电器安装板作电气连接。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "N端子板和PE端子板未分设，或其与安装板的绝缘/连接关系清晰不符合要求",
    },
    "TEMP_ELEC_WIRING_INSULATION": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.11条",
        "text": "配电箱、开关箱内的连接线必须采用铜芯绝缘导线，线束应有外套绝缘管，导线应与电器端子连接牢固，不得有外露带电部分。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "绝缘破损、线束无绝缘保护或带电部分清晰外露",
    },
    "TEMP_ELEC_WIRING_ENTRY": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.15条",
        "text": "配电箱、开关箱的进出线口应配置固定线卡，进出线应加绝缘护套并成束卡固在支架上，不得与箱体直接接触。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "进出线口无绝缘护套、无固定线卡或电缆与箱体直接接触",
    },
    "TEMP_ELEC_BOX_WEATHER_PROTECTION": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.1.16条",
        "text": "配电箱、开关箱应采取防雨、防尘措施。",
        "source": JGJ_T_46_2024_SETTING_SOURCE,
        "scope": "箱体缺少明显防雨、防尘措施且图片证据直接",
    },
    "TEMP_ELEC_BOX_LABEL": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.3.1条",
        "text": "配电箱、开关箱应有名称、用途、分路标记及系统接线图。",
        "source": JGJ_T_46_2024_USE_SOURCE,
        "scope": "箱体或箱内明显缺少名称、用途、分路标记及系统接线图",
    },
    "TEMP_ELEC_BOX_LOCK": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.3.2条",
        "text": "配电箱、开关箱箱门应配锁，并应由专人负责。",
        "source": JGJ_T_46_2024_USE_SOURCE,
        "scope": "箱门完整可见且明显无锁、无法关闭或锁具失效",
    },
    "TEMP_ELEC_BOX_CLEAN": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.3.8条",
        "text": "配电箱、开关箱内不得放置任何杂物，并应保持整洁。",
        "source": JGJ_T_46_2024_USE_SOURCE,
        "scope": "箱内放置与电气系统无关的杂物且证据清晰",
    },
    "TEMP_ELEC_UNAUTHORIZED_CONNECTION": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.3.10条",
        "text": "配电箱、开关箱内的电器配置和接线不得随意改动；熔断器熔体更换时，不得采用不符合原规格的熔体代替。",
        "source": JGJ_T_46_2024_USE_SOURCE,
        "scope": "可直接看出随意改动、违规跨接或明显不规范接线",
    },
    "TEMP_ELEC_TERMINAL_EXTERNAL_FORCE": {
        "standard": JGJ_T_46_2024_TITLE,
        "article": "第4.3.11条",
        "text": "配电箱、开关箱的进出线应避免外力作用，严禁与金属尖锐断口和强腐蚀介质接触。",
        "source": JGJ_T_46_2024_USE_SOURCE,
        "scope": "电缆进出箱受到外力或与尖锐边缘明显接触",
    },
}


def regulation_catalog_for_prompt(scene: str | None) -> str:
    """Return a compact, model-readable list without flooding the prompt."""
    if not scene or "TEMPORARY_ELECTRICITY" not in str(scene):
        return (
            "当前图片若不属于临时用电专项，citation_key 必须写 UNKNOWN；"
            "不得把本目录条款套用到其他场景。"
        )
    lines = [
        "临时用电可选已核验条款键（只能从以下键中选择，无法直接对应时写 UNKNOWN）："
    ]
    for key, item in VERIFIED_REGULATIONS.items():
        lines.append(f"- {key}：{item['scope']}；{item['article']}")
    return "\n".join(lines)


def resolve_regulation(citation_key: Any) -> dict[str, Any] | None:
    key = str(citation_key or "").strip().upper()
    return VERIFIED_REGULATIONS.get(key)


def format_regulation(citation_key: Any) -> str:
    item = resolve_regulation(citation_key)
    if not item:
        return "暂未匹配到已核验的具体条文；请人工查阅适用现行标准。"
    return (
        f"{item['standard']} {item['article']}：{item['text']} "
        f"（查询：{item['source']}）"
    )
