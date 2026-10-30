#!/usr/bin/env python3
"""Run paired full-image/crop experiments; keep failures in the denominator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 代码在 src/，仓库根目录放 .env 与 outputs/
REPO_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safety_check import original_image_size, parse_detail_crop  # noqa: E402


BASE_DIR = REPO_DIR


def summarize_runs(records: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    summary = {}
    for strategy in ("full_image_only", "full_plus_detail_crop"):
        runs = [record for record in records if record["strategy"] == strategy]
        successes = [record for record in runs if record["result"].get("status") == "success"]
        times = [record["wall_seconds"] for record in successes]
        timely = sum(record["wall_seconds"] <= budget for record in successes)
        summary[strategy] = {
            "attempts": len(runs),
            "successful_reports": len(successes),
            "successful_within_budget": timely,
            "within_budget_rate": timely / len(runs) if runs else None,
            "median_success_wall_seconds": round(statistics.median(times), 4) if times else None,
            "max_success_wall_seconds": max(times) if times else None,
        }
    return summary


def run_detection(
    image_path: Path,
    scene: str,
    crop: str,
    budget: float,
    dry_run: bool,
) -> dict[str, Any]:
    command = [
        sys.executable, str(BASE_DIR / "safety_check.py"), str(image_path),
        "--detail-crop", crop, "--json-output", "--no-save", "--show-timing",
    ]
    if scene:
        command.extend(["--scene", scene])
    if dry_run:
        command.append("--dry-run")
    environment = dict(os.environ, REQUEST_TIMEOUT_SECONDS=str(budget))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=BASE_DIR, env=environment, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=budget, check=False,
        )
        try:
            result = json.loads(completed.stdout)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")
        except ValueError:
            result = {"status": "error", "error": {
                "type": "invalid_json", "message": "Checker did not return a JSON object.",
            }}
        if completed.returncode and result.get("status") != "error":
            result = {"status": "error", "error": {
                "type": "process_error", "message": "Checker exited unsuccessfully.",
            }}
        if result.get("status") in {"success", "dry_run_ok"}:
            expected_images = 1 if crop == "off" else 2
            if result.get("vision_calls") != 1 or result.get("image_inputs") != expected_images:
                result = {"status": "error", "error": {
                    "type": "input_protocol_error",
                    "message": "Unexpected image count or vision call count.",
                }}
    except subprocess.TimeoutExpired:
        result = {"status": "error", "error": {
            "type": "comparison_deadline",
            "message": "Checker process was stopped at the experiment wall-time limit.",
        }}
    return {"wall_seconds": round(time.perf_counter() - started, 4), "result": result}


def markdown_report(payload: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "# 全图与局部图输入对照实验", "",
        f"- 图片：{payload['image_path']}",
        f"- 图片 SHA256：`{payload['image_sha256']}`",
        f"- 场景提示：{payload['scene'] or '未指定'}",
        f"- 局部图：`{payload['detail_crop']}`（整图 0–1000 坐标）",
        f"- 每组轮数：{payload['repeats']}；单次进程预算：{payload['budget_seconds']} 秒。",
        f"- 模式：{'仅本地预处理，不调用模型' if payload['dry_run'] else '真实 API 请求，无自动重试'}。",
        "- 奇数轮先全图、偶数轮先局部图，逐次运行，减少固定顺序和并发干扰。",
        "- 进程墙钟包含 Python 启动、预处理、请求、解析和终端 JSON 输出；不包含客户端上传到 HTTP API。",
        "- 失败和超时均计入尝试数；计时上限可能有少量进程终止开销。", "",
        "| 轮次 | 策略 | 状态 | 预处理/s | 模型/s | 脚本/s | 进程墙钟/s | 预算内有效报告 | 风险项或错误 |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for record in payload["records"]:
        result = record["result"]
        timing = result.get("timings", {})
        items = "; ".join(
            f"{issue.get('status')}: {issue.get('item')}"
            for issue in result.get("issues", [])
        ) or result.get("error", {}).get("message", "—")
        valid = result.get("status") == "success" and record["wall_seconds"] <= payload["budget_seconds"]
        values = [record["round"], record["strategy"], result.get("status"),
                  timing.get("preprocess", "—"), timing.get("deepseek_vision", "—"),
                  result.get("elapsed_seconds", "—"), record["wall_seconds"],
                  "是" if valid else "否", items]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(["", "## 汇总", "", "```json",
                  json.dumps(payload["summary"], ensure_ascii=False, indent=2), "```", "",
                  "## 解释范围", "",
                  "本实验不自动给准确率打分。须对照人工标注核验具体违规、对象状态、法规适用性及原图范围框。",
                  "少量重复和单张图片不能证明跨场景提升或稳定的 p95 时延；API 排队与输出长度也会改变耗时。",
                  "人工选择的局部框是诊断条件，不能作为自动选区能力的成绩。", ""])
    return "\n".join(lines)


def main() -> int:
    load_dotenv(BASE_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--scene", default="")
    parser.add_argument("--detail-crop", default="auto", help="auto or original-image x1,y1,x2,y2 in 0..1000.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "outputs" / "input_comparison"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 0 < args.budget < float("inf"):
        parser.error("Use 1..10 repeats and a finite positive budget.")
    image_path = Path(args.image).expanduser().resolve()
    crop = args.detail_crop.strip()
    try:
        image_size = original_image_size(image_path)
        if crop.lower() != "auto" and parse_detail_crop(crop, image_size) is None:
            parser.error("The experiment needs auto or a nonempty crop, not off.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    timestamp = datetime.now().astimezone().isoformat()
    output_dir = Path(args.output_dir).expanduser().resolve() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True)
    payload = {
        "experiment": "paired_image_inputs_v1", "created_at": timestamp,
        "image_path": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256((BASE_DIR / "safety_check.py").read_bytes()).hexdigest(),
        "scene": args.scene, "detail_crop": crop, "repeats": args.repeats,
        "budget_seconds": args.budget, "dry_run": args.dry_run,
        "config": {name: os.getenv(name, default) for name, default in {
            "DEEPSEEK_MODEL": "deepseek-flash", "MAX_IMAGE_EDGE": "768", "JPEG_QUALITY": "70",
            "IMAGE_DETAIL": "low", "DETAIL_CROP_MAX_EDGE": os.getenv("MAX_IMAGE_EDGE", "768"),
            "SINGLE_MODEL_MAX_TOKENS": os.getenv("MAX_TOKENS", "2200"),
            "REASONING_EFFORT": "none", "TEMPERATURE": "0",
            "DEEPSEEK_VISION_TIMEOUT_SECONDS": str(max(1.0, args.budget - 0.55)),
        }.items()},
        "records": [],
    }
    for round_number in range(1, args.repeats + 1):
        modes = [("full_image_only", "off"), ("full_plus_detail_crop", crop)]
        if round_number % 2 == 0:
            modes.reverse()
        for strategy, setting in modes:
            record = {"round": round_number, "strategy": strategy,
                      **run_detection(image_path, args.scene, setting, args.budget, args.dry_run)}
            filename = f"round-{round_number:02d}-{strategy}.json"
            (output_dir / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            payload["records"].append(record)
            print(f"round={round_number} strategy={strategy} status={record['result'].get('status')} wall={record['wall_seconds']:.3f}s", file=sys.stderr)
    payload["summary"] = summarize_runs(payload["records"], args.budget)
    (output_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "comparison.md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"status": "experiment_complete", "dry_run": args.dry_run,
                      "json_path": str(output_dir / "comparison.json"),
                      "report_path": str(output_dir / "comparison.md"),
                      "summary": payload["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
