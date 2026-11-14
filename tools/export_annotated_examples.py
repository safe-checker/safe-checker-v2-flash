#!/usr/bin/env python3
"""Export curated, judge-confirmed detections as annotated images.

This tool does not call a model. It joins an existing evaluation record with
its LLM-judge verdict, draws only the issue that matched the ground truth, and
writes an auditable manifest beside the rendered images.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_SELECTION = {
    "施工用电": ["用电10", "用电14", "补电07", "补电08"],
    "高处作业": ["高处06", "高处10", "高处11", "高处20"],
    "塔吊": ["塔吊02", "塔吊11", "塔吊20", "补塔05"],
    "脚手架": ["架体02", "架体12", "补架03", "补架06"],
    "起重吊装": ["吊装06", "吊装11", "吊装15", "补吊11"],
    "基坑工程": ["基坑03", "基坑13", "基坑15", "基坑20"],
    "施工升降机": ["升降机05", "升降机04", "升降机13", "升降机17"],
}

FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Chinese-capable font was found")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path()), size=max(size, 12))


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text.strip():
        trial = current + char
        if current and text_width(draw, trial, font) > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", value).strip("_")


def resolve_bbox(issue: dict[str, Any], image: Image.Image) -> tuple[int, int, int, int]:
    bbox = issue.get("bbox_original_px") or {}
    values = [bbox.get(name) for name in ("x", "y", "width", "height")]
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError("matched issue has no numeric bbox_original_px")
    x, y, width, height = (int(round(value)) for value in values)
    x1 = max(0, min(image.width - 1, x))
    y1 = max(0, min(image.height - 1, y))
    x2 = max(x1 + 1, min(image.width, x + width))
    y2 = max(y1 + 1, min(image.height, y + height))
    return x1, y1, x2, y2


def annotate(
    source: Path,
    target: Path,
    scene: str,
    sample_id: str,
    issue: dict[str, Any],
) -> tuple[int, int, int, int]:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

    x1, y1, x2, y2 = resolve_bbox(issue, image)
    stroke = max(5, min(image.width, image.height) // 180)
    label_size = max(28, min(64, image.width // 24))
    label_font = load_font(label_size)
    draw = ImageDraw.Draw(image)
    red = (224, 32, 32)
    draw.rectangle((x1, y1, x2, y2), outline=red, width=stroke)

    marker = "1"
    marker_box = draw.textbbox((0, 0), marker, font=label_font, stroke_width=1)
    marker_w = marker_box[2] - marker_box[0] + stroke * 4
    marker_h = marker_box[3] - marker_box[1] + stroke * 3
    marker_y = max(0, y1 - marker_h)
    draw.rounded_rectangle(
        (x1, marker_y, min(image.width, x1 + marker_w), marker_y + marker_h),
        radius=max(3, stroke),
        fill=red,
    )
    draw.text(
        (x1 + stroke * 2, marker_y + stroke),
        marker,
        font=label_font,
        fill="white",
        stroke_width=1,
        stroke_fill="white",
    )

    panel_font_size = max(24, min(48, image.width // 30))
    panel_font = load_font(panel_font_size)
    small_font = load_font(max(20, int(panel_font_size * 0.78)))
    scratch = ImageDraw.Draw(Image.new("RGB", (image.width, 100), "white"))
    margin = max(24, image.width // 50)
    text_max = image.width - margin * 2
    title_lines = wrap_text(
        scratch,
        f"{scene} | {sample_id} | 违规项：{issue.get('item', '')}",
        panel_font,
        text_max,
    )
    evidence = str(issue.get("evidence") or "").strip()
    detail_lines = wrap_text(scratch, f"图框1证据：{evidence}", small_font, text_max)
    line_h = int(panel_font_size * 1.45)
    small_line_h = int(small_font.size * 1.45)
    panel_h = margin * 2 + len(title_lines) * line_h + len(detail_lines) * small_line_h

    canvas = Image.new("RGB", (image.width, image.height + panel_h), "white")
    canvas.paste(image, (0, 0))
    panel_draw = ImageDraw.Draw(canvas)
    panel_draw.rectangle((0, image.height, image.width, image.height + panel_h), fill=(250, 250, 250))
    panel_draw.rectangle((0, image.height, max(8, stroke), image.height + panel_h), fill=red)
    cursor_y = image.height + margin
    for line in title_lines:
        panel_draw.text((margin, cursor_y), line, font=panel_font, fill=(30, 30, 30))
        cursor_y += line_h
    for line in detail_lines:
        panel_draw.text((margin, cursor_y), line, font=small_font, fill=(80, 80, 80))
        cursor_y += small_line_h

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92, subsampling=0)
    return x1, y1, x2 - x1, y2 - y1


def make_contact_sheet(paths: list[Path], target: Path, scene: str) -> None:
    thumb_w, thumb_h = 760, 570
    gap, header_h = 28, 96
    cols = 2
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * gap, header_h + rows * thumb_h + (rows + 1) * gap),
        (241, 243, 245),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 22), f"{scene}：检测命中示例（红框为违规位置）", font=load_font(38), fill=(25, 25, 25))
    for index, path in enumerate(paths):
        with Image.open(path) as raw:
            item = raw.convert("RGB")
        item.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = gap + (index % cols) * (thumb_w + gap) + (thumb_w - item.width) // 2
        y = header_h + gap + (index // cols) * (thumb_h + gap) + (thumb_h - item.height) // 2
        sheet.paste(item, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--judge", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="family_parts_b4_r3_i12")
    args = parser.parse_args()

    labels = {row["sample_id"]: row for row in load_jsonl(args.labels)}
    records = {
        row["sample_id"]: row
        for row in load_jsonl(args.records)
        if row.get("version") == args.version
    }
    verdict_rows = json.loads(args.judge.read_text(encoding="utf-8"))
    verdicts = {
        row["sample_id"]: row
        for row in verdict_rows
        if row.get("version") == args.version
    }

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    scene_outputs: dict[str, list[Path]] = defaultdict(list)

    for scene, sample_ids in DEFAULT_SELECTION.items():
        for sample_id in sample_ids:
            label = labels[sample_id]
            record = records[sample_id]
            verdict = verdicts[sample_id]
            judge = verdict.get("judge") or {}
            if not judge.get("hit") or judge.get("_unjudged"):
                raise ValueError(f"{sample_id}: selected sample is not a valid judged hit")
            issue_index = int(judge.get("index") or 0)
            issues = (record.get("result") or {}).get("issues") or []
            if not (1 <= issue_index <= len(issues)):
                raise ValueError(f"{sample_id}: invalid judge issue index {issue_index}")
            issue = issues[issue_index - 1]
            source = Path(record["image_path"])
            if not source.is_absolute():
                source = args.dataset_root / source
            filename = f"{safe_name(sample_id)}_{safe_name(str(issue.get('item') or '违规项'))}.jpg"
            target = args.out / safe_name(scene) / filename
            bbox = annotate(source, target, scene, sample_id, issue)
            scene_outputs[scene].append(target)
            manifest.append(
                {
                    "scene": scene,
                    "sample_id": sample_id,
                    "source_image": str(source),
                    "annotated_image": str(target),
                    "ground_truth": label.get("inspection_item", ""),
                    "detected_issue": issue.get("item", ""),
                    "evidence": issue.get("evidence", ""),
                    "confidence": issue.get("confidence"),
                    "bbox_original_px": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                    },
                    "judge_index": issue_index,
                    "judge_reason": judge.get("reason", ""),
                    "wall_seconds": record.get("wall_seconds"),
                }
            )

    for scene, paths in scene_outputs.items():
        make_contact_sheet(paths, args.out / f"总览_{safe_name(scene)}.jpg", scene)

    (args.out / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 各场景检测命中与违规框选示例",
        "",
        f"- 检测版本：`{args.version}`",
        "- 来源：140 张施工安全陌生测试集的真实跑测记录",
        "- 筛选条件：检测成功、LLM 裁判命中、命中项包含有效原图像素坐标框",
        "- 红框含义：只框出裁判确认命中的违规项；没有展示同图内未经裁判确认的其他输出",
        "",
    ]
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_scene[row["scene"]].append(row)
    for scene in DEFAULT_SELECTION:
        lines.extend(
            [
                f"## {scene}",
                "",
                f"![{scene}总览](总览_{safe_name(scene)}.jpg)",
                "",
                "| 样本 | 标注违规项 | 系统命中项 | 置信度 | 坐标框 `(x,y,w,h)` |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in by_scene[scene]:
            bbox = row["bbox_original_px"]
            rel = Path(row["annotated_image"]).relative_to(args.out).as_posix()
            lines.append(
                f"| [{row['sample_id']}]({rel}) | {row['ground_truth']} | "
                f"{row['detected_issue']} | {row['confidence']:.2f} | "
                f"`({bbox['x']},{bbox['y']},{bbox['width']},{bbox['height']})` |"
            )
        lines.append("")
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(manifest)} annotated images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
