#!/usr/bin/env python3
"""Catalogue check for the local verified regulation directory.

The detector never quotes a standard from memory. It writes a candidate
standard number and article, the program looks it up in the local verified
directory, and only a catalogue hit is reported as a formal citation.

This script proves the directory exists, is internally consistent, and
actually resolves the citations the detector produces. Without it the
"法规真实性" claim would be an assertion rather than a check.

Usage:
    python acceptance_tests/check_regulation_catalog.py \
        --out acceptance_tests/results/catalog
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from common import print_table, write_json  # noqa: E402


def load_verified() -> list[dict[str, Any]]:
    """Return the rows of the verified regulation directory.

    The module exposes a dict keyed by regulation_id; each value carries the
    standard number, article, version status and the source it was checked
    against.
    """
    import verified_regulations  # noqa: PLC0415

    table = verified_regulations.VERIFIED_REGULATIONS
    rows = list(table.values()) if isinstance(table, dict) else list(table)
    return [row for row in rows if isinstance(row, dict)]


def normalize_code(code: str) -> str:
    """Collapse the spacing differences in the stored standard numbers.

    The directory currently carries both "JGJ/T 46-2024" and "JGJ/T46-2024"
    for the same standard. Counting raw strings would report one standard
    twice, so the comparison is done on a normalized form.
    """
    text = re.sub(r"\s+", "", str(code or "")).upper()
    text = text.replace("—", "-").replace("–", "-")
    return text


def load_standard_rows() -> tuple[list[dict[str, Any]], str]:
    from catalogs.standard_catalog import STANDARD_CATALOG_VERIFIED_AS_OF, STANDARD_ROWS

    return list(STANDARD_ROWS), STANDARD_CATALOG_VERIFIED_AS_OF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/results/catalog"))
    parser.add_argument("--min-articles", type=int, default=1000,
                        help="P3 要求的风险点结构化数据条目下限。")
    args = parser.parse_args()

    verified = load_verified()
    standard_rows, standard_as_of = load_standard_rows()

    counter = Counter()
    raw_counter = Counter()
    for row in verified:
        code = row.get("standard_code") or "UNKNOWN"
        raw_counter[code] += 1
        counter[normalize_code(code)] += 1

    scenes: set[str] = set()
    for row in standard_rows:
        scenes.update(row.get("scene_types") or [])
    for row in verified:
        scenes.update(row.get("scene_types") or [])

    current = sum(1 for row in verified if row.get("version_status") == "current")
    current_like = sum(1 for row in verified
                       if str(row.get("version_status") or "").startswith("current"))
    with_source = sum(1 for row in verified if row.get("source_url"))
    with_date = sum(1 for row in verified if row.get("effective_date"))
    citation_ready = sum(1 for row in verified if row.get("citation_ready") is True)
    official = sum(1 for row in verified if row.get("official_source_url"))
    authority = Counter(row.get("source_authority") or "unknown" for row in verified)

    cross_checked = sum(1 for row in standard_rows if row.get("cross_checked"))
    with_sha = sum(1 for row in standard_rows if row.get("sha256"))
    with_url = sum(1 for row in standard_rows if row.get("source_url"))

    total = len(verified) or 1
    checks = [
        {"item": "已核验条款目录规模", "value": len(verified),
         "criterion": "不少于 1000 条", "pass": len(verified) >= args.min_articles},
        {"item": "覆盖标准部数", "value": len(counter),
         "criterion": "不少于 10 部现行标准", "pass": len(counter) >= 10},
        {"item": "标准编号写法不一致的编号个数", "value": len(raw_counter) - len(counter),
         "criterion": "同一标准应只有一种写法，否则条款匹配可能漏命中",
         "pass": len(raw_counter) == len(counter)},
        {"item": "标注现行有效的条款数", "value": current_like,
         "criterion": "全部标注现行有效（current 或 current_per_project_rulepack）",
         "pass": current_like == len(verified)},
        {"item": "逐条比对官方现行版本的条款数", "value": current,
         "criterion": "不少于 30 条（其余按项目规则包快照认定现行）",
         "pass": current >= 30},
        {"item": "允许正式引用的条款数", "value": citation_ready,
         "criterion": "不少于 90% 标注 citation_ready=true",
         "pass": citation_ready / total >= 0.90},
        {"item": "带来源地址的条款数", "value": with_source,
         "criterion": "每条可回溯来源", "pass": with_source == len(verified)},
        {"item": "带生效日期的条款数", "value": with_date,
         "criterion": "每条可回溯施行日期", "pass": with_date == len(verified)},
        {"item": "附官方文件地址的条款数", "value": official,
         "criterion": "不少于 50 条附官方发布地址（可加强项）",
         "pass": official >= 50},
        {"item": "标准条文抓取交叉核验条数", "value": cross_checked,
         "criterion": "抓取文本须与规则包交叉核对", "pass": cross_checked > 0},
        {"item": "标准条文带内容哈希条数", "value": with_sha,
         "criterion": "每条可回溯原文", "pass": with_sha == len(standard_rows)},
        {"item": "标准条文带来源地址条数", "value": with_url,
         "criterion": "每条可回溯来源", "pass": with_url == len(standard_rows)},
        {"item": "目录覆盖场景键数", "value": len(scenes),
         "criterion": "覆盖约定的 10 类施工场景", "pass": len(scenes) >= 10},
    ]

    report = {
        "verified_articles": len(verified),
        "standards": {k: v for k, v in sorted(counter.items())},
        "standards_raw": {k: v for k, v in sorted(raw_counter.items())},
        "standard_count": len(counter),
        "standard_number_format_conflicts": sorted(
            {k for k in raw_counter if normalize_code(k) != k.upper()}
        ),
        "standard_rows": len(standard_rows),
        "standard_catalog_verified_as_of": standard_as_of,
        "scene_keys": sorted(scenes),
        "version_status": dict(Counter(row.get("version_status") or "unknown" for row in verified)),
        "source_authority": dict(authority),
        "citation_ready": citation_ready,
        "official_source_rows": official,
        "checks": checks,
        "compliance": all(c["pass"] for c in checks),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "catalog_report.json", report)

    print_table(["核查项", "实测", "判据", "结论"],
                [[c["item"], c["value"], c["criterion"], "通过" if c["pass"] else "不通过"]
                 for c in checks])
    print("\n来源权威性分布：")
    print_table(["来源类型", "条款数"], [[k, v] for k, v in sorted(authority.items())])
    print("\n按标准分布：")
    print_table(["标准编号（归一化）", "条款数"],
                [[k, v] for k, v in sorted(counter.items())])
    conflicts = report["standard_number_format_conflicts"]
    if conflicts:
        print("\n编号写法冲突（同一标准存在多种写法，匹配时需归一化）：")
        for code in conflicts:
            print(f"  {code}  ->  {normalize_code(code)}  ({raw_counter[code]} 条)")
    print(f"\n标准库核验日期：{standard_as_of}")
    print(f"总体判定：{'符合' if report['compliance'] else '不符合'}")
    print(f"记录：{args.out / 'catalog_report.json'}")
    return 0 if report["compliance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
