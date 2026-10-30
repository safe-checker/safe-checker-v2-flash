#!/usr/bin/env python3
"""Clean the scraped standard text and drop the explanatory notes.

The standard text was scraped from web pages, so it carries three kinds of
noise that hurt rule generation:

* variant and full-width characters ("圖手架", "釆用", "∕") that no keyword
  ever matches;
* English fragments mangled by the scraper ("SCaffoH" for "Scaffold");
* the standard's own 条文说明 (explanatory notes) mixed in with the normative
  clauses. Those read like requirements but explain nothing that can be checked
  in a photograph, and turning them into rules produces rules nobody can obey.

This script writes a cleaned copy and reports what it removed, so the rule
generator can be pointed at clauses that actually say what is required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# 确定性的异体字与全角归一。只收语义等价、不会改变含义的替换。
VARIANT_CHARS = {
    "圖": "图", "釆": "采", "髙": "高", "説": "说", "査": "查", "設": "设", "専": "专",
    "驗": "验", "壓": "压", "線": "线", "級": "级", "頭": "头", "養": "养", "風": "风",
    "橋": "桥", "護": "护", "囲": "围", "対": "对", "応": "应", "価": "价", "帯": "带",
    "発": "发", "薬": "药", "廃": "废", "働": "动", "備": "备", "検": "检", "継": "继",
    "規": "规", "資": "资", "単": "单", "断": "断", "斜": "斜", "様": "样",
    "∕": "/", "〜": "~", "∽": "~", "•": "·", "‧": "·", "ˉ": "-", "‐": "-",
    "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
    "℃": "℃", "‰": "‰", "％": "%", "＃": "#", "＆": "&",
}

# 抓取时把英文术语的字形认错了。这些是确定性的还原。
OCR_TERMS = [
    (re.compile(r"SCaffoH|SCaffoId|SCaffOId|SCaffold|ScaffoH|ScaffoId", re.I), "Scaffold"),
    (re.compile(r"OPeratiOn|OPeratlOn|OPeratlon|Operation", re.I), "Operation"),
    (re.compile(r"COmPOSitiVe|Composite|COmPOSite", re.I), "Composite"),
    (re.compile(r"Safety\s*factor", re.I), "Safety factor"),
    (re.compile(r"model\s*test", re.I), "model test"),
]

# 条文说明的判据：叙述语气 + 异常长度 + 目录行
# 注意："宜按"、"参照本标准" 是规范里的正常表述（表示推荐做法），
# 不能当叙述语气，否则会把"宜按减小步距加密设置"这类可检查条款误杀。
# 长度阈值只用来兜住前言、附录、符号表这类整段搬运，不承担判断职责——
# 判断"这条能不能查"是生成器和校验器的活，这里宁可放过。
NARRATIVE = re.compile(
    r"经调查|经分析|本条文|本章条文|条文所述|本章规定|其含义|理解与执行|"
    r"规范编制组|本规范修订|本规范的主要技术内容|本章的编制|"
    r"根据.{0,12}调查|本章条文说明|应按.{0,6}理解"
)
# 目录行的特征是"数字+短词"连续排布且中间没有任何句读；
# 列项（"1施工荷载；2风荷载"）带分号，不能算目录行。
TOC_LINE = re.compile(r"(?:\d{1,2}(?:\.\d{1,2})?\s*[^\d\s。；，、）)]{2,12}\s*){3,}")
LONG_CLAUSE = 800


def clean_text(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    out = str(text or "")
    for source, target in OCR_TERMS:
        out, count = source.subn(target, out)
        if count:
            notes.append(f"ocr_term:{count}")
    for source, target in VARIANT_CHARS.items():
        if source in out:
            out = out.replace(source, target)
            notes.append(f"variant:{source}")
    # 挤在一起的章节标题行，例如 "8 构造要求 8.1 一般规定 8.2 作业脚手架"
    out = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d{1,2}\s)", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(), notes


def is_explanatory(text: str) -> tuple[bool, str]:
    """Decide whether a clause is 条文说明 rather than a normative requirement."""
    body = str(text or "")
    if not body:
        return False, "empty"
    reasons = []
    if NARRATIVE.search(body):
        reasons.append("叙述语气")
    if len(body) > LONG_CLAUSE:
        reasons.append(f"长度{len(body)}")
    toc = TOC_LINE.search(body)
    if toc and not re.search(r"[。；，、]", toc.group(0)):
        reasons.append("目录行")
    return bool(reasons), "、".join(reasons)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(BASE / "resources" / "standards" / "cleaned_clauses.json"))
    parser.add_argument("--report", default=str(BASE / "resources" / "standards" / "clean_report.json"))
    args = parser.parse_args()

    from standard_catalog import STANDARD_ROWS

    kept, dropped = [], []
    touched = 0
    for row in STANDARD_ROWS:
        cleaned, notes = clean_text(row.get("clause_text"))
        if notes:
            touched += 1
        explanatory, reason = is_explanatory(cleaned)
        record = dict(row)
        record["clause_text"] = cleaned
        record["clean_notes"] = notes
        if explanatory:
            record["explanatory_reason"] = reason
            dropped.append(record)
        else:
            kept.append(record)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {
        "source": "standard_catalog.STANDARD_ROWS",
        "total": len(STANDARD_ROWS),
        "kept": len(kept),
        "dropped_explanatory": len(dropped),
        "text_modified": touched,
        "dropped_detail": [
            {"regulation_id": r["regulation_id"], "standard_code": r["standard_code"],
             "article": r["article"], "reason": r["explanatory_reason"],
             "chars": len(r["clause_text"])}
            for r in dropped
        ],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"总条款 {len(STANDARD_ROWS)}")
    print(f"  清洗后保留 {len(kept)}")
    print(f"  识别为条文说明并剔除 {len(dropped)}")
    print(f"  文本被修正的条款 {touched}")
    for item in report["dropped_detail"][:12]:
        print(f"    剔除 {item['standard_code']} {item['article']}  {item['chars']}字  ({item['reason']})")
    print(f"写入 {out} 与 {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
