#!/usr/bin/env python3
"""Turn standard clauses into rules that can drive a check.

The rule pack gives us 63 hand-curated rules; the standards give us 1081 clauses.
Most of those clauses are not rules yet: they have text and a number, but no
statement of what to look at, what counts as a violation, and where in the photo
the subject would be. This script asks a model to write those parts down, using
the same methodology the full checker uses: a clause is turned into rule
*definitions*, never into a verdict about a particular photo.

The output is a candidate pool. Nothing here reaches the prompt automatically;
see tools/validate_generated_rules.py for the gates a rule must pass.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from dotenv import load_dotenv

load_dotenv(BASE / ".env")

SYSTEM_PROMPT = """你是【施工安全规范条款转可执行规则】专家。

你的任务：根据输入的规范条款原文和上下文，把它转写成可执行的检查规则。
你输出的是**规则定义**，不是对某张现场照片的裁决结论。

## 执行步骤
1. 先判断这条条款是不是容器性/引言性条款（只讲原则、适用范围、术语定义，没有可检查的对象）。
   是则输出空数组。
2. 若是系统关系、台账资料、检测记录、审批流程、资质证书、承载力验算、接地电阻实测这类条款，
   静态照片判不了，也输出空数组，并把理由写进 note。
3. 否则先判断适用性，再决定要不要拆成多条 atomic rule。
4. 每条 atomic rule 写清：判断对象、违规边界、合规边界、存疑边界、适用前提、
   证据要求、消歧说明、检索关键词。
5. 输出一个合法 JSON 对象。

## 关键业务规则
1. **只依据可见、可复核的证据裁决。** non_compliant 只描述"已经被看得见的证据直接支撑的违规事实"，
   不要把推测、延伸推断、不可见内部状态、被遮挡区域写成违规。
2. compliant 是"在规则所需的可见判断对象上，未见违规时的成立边界"，
   它是规则边界定义，**不是对看不见的部位作正向背书**。
3. doubtful 只用于"同一可见判断对象存在直接冲突、关键可见特征互相矛盾"；
   **不要把"没拍到 / 看不见 / 遮挡"直接写成 doubtful**。
4. 条款涉及系统关系、隐蔽部位时，不要因此返回空规则，
   而是把边界改写成"只有在相应证据被看见时才可判违规"。
5. 条款边界要清晰，不要把相邻条款的内容混进来。
6. judge_dimension 只说本规则判断什么、不判断什么，一到两句。
7. **规则要能落到照片里的一个具体部位或对象上。**
   如果写不出"在图里看哪儿"，这条就不该是规则。

## 输出格式
只输出一个合法 JSON 对象，不要输出解释文本或 Markdown。
{
  "rules": [
    {
      "atomic_rule_id": "由条款号和短标识拼成，例如 GB51210-2016-6.2.3@guardrail",
      "clause_text": "面向裁决的条款表述，尽量忠实原文",
      "rule_family": "visual_direct|visual_structural|system_topology|ledger_only|hybrid",
      "judge_dimension": "本规则判断的对象、关系或状态",
      "judge_boundary": {
        "compliant": "未见可复核违规事实的边界",
        "non_compliant": "已被可见证据直接支撑的违规边界",
        "doubtful": "仅在同一可见判断对象存在直接冲突时使用"
      },
      "applicability": {"all_of": ["前置条件"], "not_applicable_when": ["不适用条件"]},
      "evidence_requirements": ["需要的可见证据"],
      "disambiguation_notes": ["容易误判的地方"],
      "keywords": ["检索关键词"],
      "similar_words": ["同义表述"],
      "scene_hint": "最相关的场景键，没有就写 GENERAL_SITE",
      "subject_hint": "规则检查的主体名称（如 作业层外侧临空边缘、配电箱开关箱及周边）",
      "subject_cue": "这个主体在照片里长什么样，怎么认出来（不超过30字）",
      "note": "留空或写一句说明"
    }
  ]
}
"""

USER_TEMPLATE = """标准：{standard_code} {standard_name}
条款号：{article}
条款原文：
{clause_text}

{context}
请输出该条款对应的规则 JSON。"""


def build_context(row: dict, by_article: dict) -> str:
    """Parent clause and neighbouring clauses, so the model keeps the hierarchy."""
    article = str(row.get("article") or "")
    parts = []
    match = re.match(r"第([\d.]+?)条", article)
    if match:
        number = match.group(1)
        # 抓取到的文本通常只有章级（第6条）和条级（第6.2.1条），
        # 中间的节级（第6.2条）往往没有单独成条，所以逐级往上找。
        parts_number = number.split(".")
        while len(parts_number) > 1:
            parts_number = parts_number[:-1]
            parent = ".".join(parts_number)
            parent_row = by_article.get((row.get("standard_code"), f"第{parent}条"))
            if parent_row:
                parts.append(f"上级条款（第{parent}条）原文：{str(parent_row.get('clause_text'))[:220]}")
                break
    parts.append("场景提示：该标准适用的场景为 " + "、".join(row.get("scene_types") or []))
    return "\n".join(parts)


async def one(client, sem, row, context, model, timeout):
    prompt = USER_TEMPLATE.format(
        standard_code=row.get("standard_code"),
        standard_name=row.get("standard_name"),
        article=row.get("article"),
        clause_text=str(row.get("clause_text") or "")[:900],
        context=context,
    )
    async with sem:
        try:
            response = await client.post(
                os.environ["DEEPSEEK_BASE_URL"],
                headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1400,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                },
                timeout=timeout,
            )
            text = response.json()["choices"][0]["message"].get("content") or ""
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return {"regulation_id": row.get("regulation_id"), "article": row.get("article"),
                    "standard_code": row.get("standard_code"), "ok": True,
                    "rules": data.get("rules") or [], "note": data.get("note") or ""}
        except Exception as exc:
            return {"regulation_id": row.get("regulation_id"), "article": row.get("article"),
                    "standard_code": row.get("standard_code"), "ok": False,
                    "rules": [], "error": f"{type(exc).__name__}: {exc}"}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", default="", help="Only this standard code, e.g. 'GB 51210-2016'.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N clauses.")
    parser.add_argument("--out", default=str(BASE / "resources" / "generated_rules_raw.json"))
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    # 读清洗后的条款：异体字已归一、条文说明已剔除
    cleaned_path = BASE / "resources" / "standards" / "cleaned_clauses.json"
    if cleaned_path.exists():
        rows = list(json.loads(cleaned_path.read_text(encoding="utf-8")))
    else:
        from standard_catalog import STANDARD_ROWS
        rows = list(STANDARD_ROWS)
    if args.standard:
        rows = [r for r in rows if r.get("standard_code") == args.standard]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("没有匹配的条款", file=sys.stderr)
        return 1

    by_article = {(r.get("standard_code"), r.get("article")): r for r in rows}
    import httpx

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        for item in json.loads(out_path.read_text(encoding="utf-8")):
            done[item["regulation_id"]] = item
    todo = [r for r in rows if r.get("regulation_id") not in done]
    print(f"待生成 {len(todo)} 条（已有 {len(done)}）", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = [one(client, sem, row, build_context(row, by_article), args.model, args.timeout)
                 for row in todo]
        finished = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            done[result["regulation_id"]] = result
            finished += 1
            if finished % 20 == 0 or finished == len(todo):
                rules = sum(len(v.get("rules") or []) for v in done.values())
                print(f"  {finished}/{len(todo)} 条款，累计 {rules} 条规则，"
                      f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
                out_path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    out_path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    rules = sum(len(v.get("rules") or []) for v in done.values())
    errors = sum(1 for v in done.values() if not v.get("ok"))
    print(f"完成：{len(done)} 个条款 -> {rules} 条规则，{errors} 个条款失败，写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
