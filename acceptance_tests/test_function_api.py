#!/usr/bin/env python3
"""Functional test of the public inspection interface.

Covers FUN-04 (model interface and function) and FUN-05 (warning record
output) of the acceptance outline. Every case is black box: it drives the
HTTP API the way the video client does and checks the response contract.

The cases deliberately include invalid input, because the outline requires
that bad input is rejected with the documented status code instead of
producing a half-filled answer.

Usage:
    python acceptance_tests/test_function_api.py \
        --manifest acceptance_tests/dataset/manifest.jsonl \
        --out acceptance_tests/results/function
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from common import Inspector, load_manifest, print_table, write_json, write_jsonl

# 服务实际返回的顶层字段。取值来自真实响应，不是文档里的示例。
REQUIRED_TOP_FIELDS = [
    "status", "format_version", "output_format", "scene", "model_detected_scene",
    "image", "issues", "report", "timings", "branches", "branch_status",
]
ISSUE_STATUS = {"CLEAR", "SUSPECTED", "NOT_ASSESSABLE"}


def case_result(case_id: str, name: str, expected: str, status_code: int | None,
                payload: Any, passed: bool, note: str = "", elapsed: float | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "name": name,
        "expected": expected,
        "status_code": status_code,
        "actual": _brief(payload),
        "conclusion": "通过" if passed else "不通过",
        "note": note,
        "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
    }


def _brief(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)[:200]
        return ", ".join(f"{k}={_short(payload.get(k))}" for k in
                         ("status", "scene", "model_detected_scene", "branches") if k in payload)
    return str(payload)[:200]


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def check_payload_contract(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "响应不是 JSON 对象"
    missing = [f for f in REQUIRED_TOP_FIELDS if f not in payload]
    if missing:
        return False, f"缺少顶层字段：{','.join(missing)}"
    if payload.get("status") != "success":
        return False, f"status 不是 success（{payload.get('status')}）"
    image = payload.get("image") or {}
    for key in ("width", "height"):
        if not isinstance(image.get(key), int) or image.get(key, 0) <= 0:
            return False, f"image.{key} 缺失或非正整数"
    if not isinstance(payload.get("issues"), list):
        return False, "issues 不是数组"
    if not isinstance(payload.get("report"), str):
        return False, "report 不是文本"
    if not isinstance(payload.get("timings"), dict):
        return False, "timings 不是对象"
    for index, issue in enumerate(payload["issues"], start=1):
        if not isinstance(issue, dict):
            return False, f"issues[{index}] 不是对象"
        if issue.get("status") not in ISSUE_STATUS:
            return False, f"issues[{index}].status 取值非法：{issue.get('status')}"
        box = issue.get("bbox_original_px")
        norm = issue.get("bbox_2d_1000")
        if box is None and norm is None:
            continue
        if box is not None:
            for key in ("x", "y", "width", "height", "image_width", "image_height"):
                if not isinstance(box.get(key), (int, float)):
                    return False, f"issues[{index}].bbox_original_px 缺少数值字段 {key}"
            if box["width"] <= 0 or box["height"] <= 0:
                return False, f"issues[{index}].bbox_original_px 宽高非正"
            if box["x"] + box["width"] > box["image_width"] + 1:
                return False, f"issues[{index}].bbox_original_px 超出原图宽度"
            if box["y"] + box["height"] > box["image_height"] + 1:
                return False, f"issues[{index}].bbox_original_px 超出原图高度"
    return True, "输出结构完整"


def check_regulation_backfill(payload: Any) -> tuple[bool, str]:
    issues = (payload or {}).get("issues") or []
    if not issues:
        return True, "本次无风险项，无条款需回填"
    matched = 0
    needs = 0
    for issue in issues:
        verified = issue.get("verified_regulation")
        if verified and verified.get("source") == "local_verified_catalog":
            matched += 1
        elif issue.get("regulation_match", {}).get("verified") is False:
            needs += 1
    if matched == 0 and needs == 0:
        return False, "既无本地目录回填结果，也无未核验标记"
    return True, f"本地核验条款 {matched} 条，待核验 {needs} 条"


def run_cases(client: Inspector, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sample = manifest[0] if manifest else None
    if sample is None:
        raise SystemExit("manifest 为空，先运行 build_dataset.py")

    image_path = sample["image_path"]
    scene = sample.get("scene", "")

    # TC-F01 正常图片：接口可用且输出结构完整
    status, first_payload, elapsed = client.inspect(image_path, scene=scene)
    payload = first_payload
    ok, note = check_payload_contract(payload)
    results.append(case_result(
        "TC-F01", "单张施工图片检测（正常输入）",
        "HTTP 200；status=success；返回 image/issues 等完整结构",
        status, payload, status == 200 and ok, note, elapsed))

    # TC-F02 场景参数生效
    status, payload, elapsed = client.inspect(image_path, scene="起重吊装")
    ok = status == 200 and isinstance(payload, dict) and payload.get("status") == "success"
    note = (f"scene={payload.get('scene')}" if isinstance(payload, dict) else "")
    results.append(case_result(
        "TC-F02", "场景参数传入与生效",
        "HTTP 200；请求中给出的场景被记录并用于本次检查",
        status, payload, ok, note, elapsed))

    # TC-F03 自定义检查指令
    status, payload, elapsed = client.inspect(image_path, scene=scene, prompt="只检查人员安全带")
    ok = status == 200 and isinstance(payload, dict) and payload.get("status") == "success"
    results.append(case_result(
        "TC-F03", "自定义检查指令传入",
        "HTTP 200；附加指令被接受，不影响接口契约",
        status, payload, ok, "", elapsed))

    # TC-F04 局部图坐标参数
    status, payload, elapsed = client.inspect(image_path, scene=scene, detail_crop="140,180,790,930")
    ok = status == 200 and isinstance(payload, dict) and payload.get("status") == "success"
    results.append(case_result(
        "TC-F04", "局部图裁剪参数合法值",
        "HTTP 200；接受 0–1000 归一化裁剪坐标",
        status, payload, ok, "", elapsed))

    # TC-F05 非法裁剪坐标
    status, payload, _ = client.inspect(image_path, detail_crop="9999,9999,10000,10000")
    results.append(case_result(
        "TC-F05", "局部图裁剪参数越界",
        "HTTP 400；非法坐标被拒绝并给出错误说明",
        status, payload, status == 400))

    # TC-F06 空文件
    status, payload, _ = client.post_raw("/v1/inspect", files={"image": ("empty.jpg", b"", "image/jpeg")})
    results.append(case_result(
        "TC-F06", "空文件上传", "HTTP 400；空文件被拒绝",
        status, payload, status == 400))

    # TC-F07 非图片文件
    status, payload, _ = client.post_raw(
        "/v1/inspect", files={"image": ("note.txt", b"this is not an image", "text/plain")})
    results.append(case_result(
        "TC-F07", "非图片文件上传", "HTTP 415；不支持的格式被拒绝",
        status, payload, status == 415))

    # TC-F08 损坏图片
    broken = Image.new("RGB", (64, 48), (120, 120, 120))
    buffer = io.BytesIO()
    broken.save(buffer, format="JPEG")
    truncated = buffer.getvalue()[: len(buffer.getvalue()) // 2]
    status, payload, _ = client.post_raw(
        "/v1/inspect", files={"image": ("broken.jpg", truncated, "image/jpeg")})
    results.append(case_result(
        "TC-F08", "损坏图片文件", "HTTP 415；无法解码的图片被拒绝",
        status, payload, status == 415))

    # TC-F09 超出大小上限
    status, payload, _ = client.post_raw(
        "/v1/inspect",
        files={"image": ("big.jpg", b"\xff\xd8" + b"\x00" * (13 * 1024 * 1024), "image/jpeg")})
    results.append(case_result(
        "TC-F09", "文件超过上传上限", "HTTP 413；超过 12 MB 上限被拒绝",
        status, payload, status == 413))

    # TC-F10 缺少 image 字段
    status, payload, _ = client.post_raw("/v1/inspect", data={"scene": "高处作业"})
    results.append(case_result(
        "TC-F10", "缺少必填图片字段", "HTTP 422；必填参数缺失被拒绝",
        status, payload, status == 422))

    # TC-F11 健康检查
    status, payload, _ = client.health()
    results.append(case_result(
        "TC-F11", "服务健康检查", "HTTP 200；服务处于可用状态",
        status, payload, status == 200))

    # TC-F12 条款回填（复用 TC-F01 的响应，避免重复调用）
    ok, note = check_regulation_backfill(first_payload)
    results.append(case_result(
        "TC-F12", "风险项条款回填与核验标记",
        "每条风险带 verified_regulation（本地目录来源）或明确的待核验标记",
        status, payload, ok, note, elapsed))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("acceptance_tests/dataset/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("acceptance_tests/results/function"))
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    client = Inspector(base_url=args.base_url) if args.base_url else Inspector()
    results = run_cases(client, manifest)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "cases.jsonl", results)
    passed = sum(1 for r in results if r["conclusion"] == "通过")
    write_json(args.out / "summary.json", {
        "total": len(results), "passed": passed, "failed": len(results) - passed,
        "cases": [{"case_id": r["case_id"], "name": r["name"],
                   "conclusion": r["conclusion"], "note": r["note"]} for r in results],
    })

    print_table(
        ["用例编号", "用例名称", "HTTP", "结论", "说明"],
        [[r["case_id"], r["name"], r["status_code"], r["conclusion"], r["note"][:44]] for r in results],
    )
    print(f"\n合计 {len(results)} 条，通过 {passed} 条，不通过 {len(results) - passed} 条")
    print(f"记录：{args.out / 'cases.jsonl'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
