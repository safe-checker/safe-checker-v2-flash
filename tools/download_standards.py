#!/usr/bin/env python3
"""Download the standards used by the rule pack and extract their clauses.

Only text that is actually present in a downloaded source is kept. Each stored
standard records the source URL, the SHA256 of the downloaded bytes and the
extraction method, so every clause can be traced back to a retrievable source.
Nothing is written by hand: a standard that cannot be retrieved or that fails
cross-checking against the rule pack is reported as pending.
"""
from __future__ import annotations

import hashlib
import html as html_mod
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "standards"
OUT.mkdir(parents=True, exist_ok=True)
CANDIDATES = OUT / "source_candidates.json"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9",
           "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}

ARTICLE_RE = re.compile(r"^\s*(\d{1,2}(?:[.．]\s*\d{1,2}){0,3})[.．]?\s*(\S.{8,900}?)\s*$")
DOT_LEADER = re.compile(r"[.·•]{6,}")
# 公告中的“……条为强制性条文，必须严格执行”清单，以及标准前言/目次说明
FORCE_LIST_RE = re.compile(r"条为强制性条文|必须严格执行|为强制性条文")
ANNOUNCEMENT_RE = re.compile(r"^\s*[、,，]?\s*\d[\d.\s、]*条?\s*(为)?强制性条文")
# 标准前言、修订说明、目次等不是条文
PREFACE_RE = re.compile(r"修订说明|住房和城乡建设部负责|本规范由|本标准由|中华人民共和国住房|目次|前言")
CHINESE = re.compile(r"[一-鿿]")
EXPLANATION_MARKERS = ("条文说明", "本标准用词说明", "引用标准名录", "本标准附录")


def fetch(url: str, timeout: int = 30, retries: int = 2) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.getcode(), response.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise last if last else RuntimeError("unknown error")


def html_to_text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030", "ignore")
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</tr>|</h\d>|</td>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html_mod.unescape(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text)


def pdf_to_text(raw: bytes) -> str:
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                chunks.append("")
    return "\n".join(chunks)


def normalize_article(number: str) -> str:
    return re.sub(r"[^0-9.]", "", number.replace("．", "."))


def parse_lines(body_lines: list[str]) -> list[dict[str, str]]:
    """Turn standard body lines into article records.

    Handles article number and text on the same line, article number alone with
    the text on the next line, and clause text continued on following lines.
    """
    clauses: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(body_lines):
        match = ARTICLE_RE.match(line)
        if match:
            number = normalize_article(match.group(1))
            body = re.sub(r"\s+", " ", match.group(2)).strip()
            if len(body) < 10 and index + 1 < len(body_lines):
                nxt = re.sub(r"\s+", " ", body_lines[index + 1]).strip()
                if len(nxt) >= 10 and CHINESE.search(nxt):
                    body = nxt
            body = re.sub(r"^(%s)\s+" % re.escape(number), "", body).strip()
            if len(body) < 10 or not CHINESE.search(body) or DOT_LEADER.search(body):
                continue
            if FORCE_LIST_RE.search(body) or ANNOUNCEMENT_RE.search(body) or PREFACE_RE.search(body):
                continue  # 公告清单、前言、修订说明都不是条文本身
            if number in seen:
                continue
            seen.add(number)
            clauses.append({"article": number, "text": body})
            continue
        if not clauses:
            continue
        piece = re.sub(r"\s+", " ", line).strip()
        if len(piece) >= 4 and CHINESE.search(piece) and not DOT_LEADER.search(piece):
            clauses[-1]["text"] = (clauses[-1]["text"] + piece).strip()
    clauses.sort(key=lambda c: [int(x) for x in c["article"].split(".")])
    return clauses


def parse_clauses(text: str) -> list[dict[str, str]]:
    """Parse the main body of a standard, skipping the 条文说明 section."""
    lines = text.split("\n")
    candidates = [len(lines)]
    for index, line in enumerate(lines):
        stripped = line.strip()
        clean = re.sub(r"[\s.·•…]+", "", stripped)
        if clean == "条文说明" or (clean.startswith("条文说明") and len(clean) <= 12):
            candidates.append(index)
        elif any(marker in stripped for marker in EXPLANATION_MARKERS) and len(stripped) < 30:
            candidates.append(index)
    best_lines = lines
    best_count = -1
    for cut in sorted(set(candidates), reverse=True):
        parsed = parse_lines(lines[:cut])
        if len(parsed) > best_count:
            best_count = len(parsed)
            best_lines = lines[:cut]
    return parse_lines(best_lines)


def main() -> int:
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    report = []
    for code, info in candidates.items():
        slug = re.sub(r"[^0-9A-Za-z]+", "-", code).strip("-")
        record: dict[str, object] = {"standard_code": code, "standard_name": info["name"],
                                      "status": "pending", "attempts": [], "clause_count": 0}
        for url in info["urls"]:
            try:
                status, raw = fetch(url)
            except Exception as exc:  # noqa: BLE001
                record["attempts"].append({"url": url, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
                continue
            sha = hashlib.sha256(raw).hexdigest()
            is_pdf = raw[:4] == b"%PDF"
            method = "pdf" if is_pdf else "html"
            try:
                text = pdf_to_text(raw) if is_pdf else html_to_text(raw)
            except Exception as exc:  # noqa: BLE001
                record["attempts"].append({"url": url, "bytes": len(raw), "sha256": sha,
                                           "method": method, "error": f"extract: {type(exc).__name__}"})
                continue
            clauses = parse_clauses(text)
            code_key = re.sub(r"[^0-9A-Za-z]", "", code).upper()
            page_key = re.sub(r"[^0-9A-Za-z]", "", text).upper()
            relevant = code_key in page_key
            record["attempts"].append({"url": url, "status": status, "bytes": len(raw),
                                       "sha256": sha, "method": method,
                                       "text_chars": len(text), "clauses_parsed": len(clauses),
                                       "relevant": relevant})
            if relevant and len(clauses) >= 15:
                (OUT / f"{slug}.json").write_text(json.dumps({
                    "standard_code": code, "standard_name": info["name"],
                    "source_url": url, "sha256": sha, "method": method,
                    "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "clause_count": len(clauses), "clauses": clauses}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                (OUT / f"{slug}.txt").write_text(text, encoding="utf-8")
                record.update(status="ok", clause_count=len(clauses), source_url=url,
                              sha256=sha, method=method, saved=str(OUT / f"{slug}.json"))
                break
        report.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "attempts"}, ensure_ascii=False))
    (OUT / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {sum(1 for r in report if r['status'] == 'ok')}/{len(report)} 部标准", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
