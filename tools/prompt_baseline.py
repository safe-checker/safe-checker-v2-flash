#!/usr/bin/env python3
"""Record the prompt baseline so parallel work streams do not overwrite each other.

The prompt is split into named blocks. Each work stream appends only inside its
own block; this script writes the current hash of every block so a change
elsewhere is visible in review.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE))
import safety_check as sc  # noqa: E402
import scene_checklists  # noqa: E402

OUT = BASE / "docs" / "prompt_baseline.json"

# 各条线只能在自己负责的块内追加内容
BLOCK_OWNERS = {
    "role_and_scene": "主控",
    "inspection_requirements": "A 对象清单线",
    "image_input_note": "主控",
    "object_state_protocol": "主控",
    "person_protocol": "主控",
    "checklist": "D 检查清单线",
    "bounding_box": "主控",
    "regulation": "B 条款回填线",
    "risk_keys": "主控",
    "status_definition": "主控",
    "output_format": "C 关系线索线 / 主控",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    body = sc._semantic_prompt_body("悬挑式脚手架")
    tail = sc._OUTPUT_TAIL_COMPACT
    blocks = {}
    for name, text in (("semantic_body", body), ("output_format_compact", tail),
                       ("output_format_full", sc._OUTPUT_TAIL_FULL)):
        blocks[name] = {"sha256_16": digest(text), "chars": len(text)}
    checklist = scene_checklists.SCENE_CHECKLISTS["SCAFFOLD_CANTILEVER"]
    payload = {
        "note": "提示词基线指纹。改提示词后重跑本脚本并提交差异，便于各条线核对是否越界修改。",
        "blocks": blocks,
        "checklist": {
            "enabled_scenes": scene_checklists.ENABLED_SCENES,
            "scantilever_dimensions": checklist["dimension_count"],
            "scantilever_chars": sum(len(x) for x in checklist["lines"]),
            "sha256_16": digest("\n".join(checklist["lines"])),
        },
        "block_owners": BLOCK_OWNERS,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
