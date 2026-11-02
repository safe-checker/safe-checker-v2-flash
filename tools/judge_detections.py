#!/usr/bin/env python3
"""Judge detection records against the ground truth with an LLM.

Lexical matching cannot tell a hit from a miss: the model writes "配电箱门敞开"
where the answer is "配电箱主门未锁闭", and a string comparison calls that a
miss. The judge therefore reads both texts and decides whether the reported
risk is the same physical problem as the labelled one.

Usage:
    python tools/judge_detections.py <records.jsonl> <labels.jsonl> [--out verdicts.json]

Every record is judged independently, so one pass judges several arms with the
same ruler. The rubric is printed with the verdicts to keep the grading
auditable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """你是施工安全检测的评审员。下面给出图片的"标准答案缺陷"和"模型报告的风险项"。
判断模型报告里是否至少有一条指向同一个安全问题。

判定口径：
1. 命中：同一主体、同一部位、同一类隐患。措辞不同、详略不同、术语不同都不影响命中。
   例如标准答案"开关箱内熔断器松动脱落，未固定完善"，模型报告"箱内断路器接线端子未固定、松动"属于命中。
2. 模型报告了另一个确实存在的问题，但与标准答案不是同一个隐患，不算命中。
3. 模型报告的问题比标准答案更严重、包含标准答案的隐患，算命中。
4. 模型只报告了疑似/无法判断的项，且没有明确指出标准答案的隐患，算不命中。
5. 模型报告了标准答案隐患所在位置，但说的是"未见缺失""合规"，不算命中。
6. 标准答案本身描述不清，只要模型指出了同一对象的同类隐患，算命中。

只输出一个 JSON 对象，不要解释：
{"hit": true 或 false, "index": 命中项的序号（从1开始，不命中填 null）, "reason": "一句话理由"}

标准答案缺陷：
{gt}

模型报告的风险项：
{issues}
"""


def build_issues(issues: list[dict[str, Any]] | None) -> str:
    """Render the detected issues into one readable block."""
    lines: list[str] = []
    for index, issue in enumerate(issues or [], start=1):
        if not isinstance(issue, dict):
            continue
        status = str(issue.get("status") or "").upper()
        item = str(issue.get("item") or issue.get("violation") or "").strip()
        evidence = str(issue.get("evidence") or "").strip()
        target = str(issue.get("target") or "").strip()
        extra = " ".join(part for part in (f"对象={target}" if target else "", evidence) if part)
        lines.append(f"{index}. 状态={status} 违规项={item} 证据={extra}".rstrip())
    return "\n".join(lines) if lines else "（模型没有报告任何风险项）"


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge returned no JSON: {text[:200]!r}")
    return json.loads(text[start : end + 1])


async def _judge_one(client: httpx.AsyncClient, url: str, headers: dict[str, str],
                     model: str, item: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.replace("{gt}", item["gt"]).replace("{issues}", item["issues"])
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    async with semaphore:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
            verdict = _extract_json(content)
            return {"hit": bool(verdict.get("hit")), "index": verdict.get("index"),
                    "reason": str(verdict.get("reason") or "")[:300]}
        except Exception as exc:  # a judge failure must not look like a detection failure
            return {"hit": False, "index": None, "reason": f"裁判调用失败：{exc}"[:300],
                    "_judge_error": type(exc).__name__}


async def judge_many(items: list[dict[str, Any]], concurrency: int = 12) -> list[dict[str, Any]]:
    load_dotenv(BASE / ".env")
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(
            *(_judge_one(client, url, headers, model, item, semaphore) for item in items)
        )


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        entries = row.get("labels") or []
        text = entries[0].get("label") if entries else row.get("inspection_item", "")
        labels[row["sample_id"]] = str(text or "")
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge detection records with an LLM.")
    parser.add_argument("records")
    parser.add_argument("labels")
    parser.add_argument("--out", default="")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--versions", default="",
                        help="只判这些版本，逗号分隔。默认全判。")
    args = parser.parse_args()

    wanted = {name.strip() for name in args.versions.split(",") if name.strip()}

    records_path = Path(args.records)
    labels = load_labels(Path(args.labels))
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    items: list[dict[str, Any]] = []
    for record in records:
        result = record.get("result") or {}
        if result.get("status") != "success":
            continue
        sample_id = record["sample_id"]
        if sample_id not in labels:
            continue
        if wanted and str(record.get("version") or "") not in wanted:
            continue
        items.append({
            "sample_id": sample_id,
            "version": record.get("version", ""),
            "gt": labels[sample_id],
            "issues": build_issues(result.get("issues")),
        })
    print(f"待判定 {len(items)} 组", file=sys.stderr)

    verdicts = asyncio.run(judge_many(items, concurrency=args.concurrency))
    out = [{**item, "judge": verdict} for item, verdict in zip(items, verdicts)]
    target = Path(args.out) if args.out else records_path.parent / "judge_verdicts.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = sum(1 for row in out if row["judge"].get("hit"))
    print(f"命中 {hits}/{len(out)}，写入 {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
