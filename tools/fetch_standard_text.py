#!/usr/bin/env python3
"""Fetch full clause text for the standards used by the rule pack.

Search driven: find candidate full-text pages, download them, parse them into
clause records and keep the candidate that yields the most well formed
clauses. Every stored text keeps its source URL, SHA256 and fetch time, so a
clause can always be traced back to where it came from. Nothing is written by
hand: if no source can be retrieved the standard is reported as pending.
"""
from __future__ import annotations

import base64
import hashlib
import html as html_mod
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "standards"
OUT.mkdir(parents=True, exist_ok=True)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9",
           "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}

ARTICLE_RE = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\s+(\S.{6,800}?)\s*$")
NOISE_HOSTS = ("bing.com", "microsoft", "baidu.com", "google.com", "so.com", "sogou.com",
               "zhihu.com", "baike", "csdn", "cnki", "docin", "doc88", "wenku", "360doc",
               "sohu.com", "163.com", "qq.com", "toutiao", "bilibili", "weixin", "mp.weixin")

# 这些站点以收录工程建设标准全文为主，优先级高于通用搜索结果
PREFERRED_HOSTS = ("mohurd.gov.cn", "jianzhu.gov.cn", "gov.cn", "ccsn.org.cn",
                   "jianzhub.cn", "gb99.com.cn", "shejiben.com", "zjt.*.gov.cn")


def fetch(url: str, timeout: int = 20, retries: int = 2) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.getcode(), response.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.8 * (attempt + 1))
    raise last if last else RuntimeError("unknown fetch error")


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


def parse_clauses(text: str) -> list[dict[str, str]]:
    clauses: list[dict[str, str]] = []
    seen: set[str] = set()
    for number, body in ARTICLE_RE.findall(text):
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < 8 or number in seen:
            continue
        if not re.search(r"[一-鿿]", body):
            continue
        seen.add(number)
        clauses.append({"article": number, "text": body})
    return clauses


def decode_bing(url: str) -> str | None:
    match = re.search(r"[?&]u=a1([^&]+)", url)
    if not match:
        return None
    payload = urllib.parse.unquote(match.group(1))
    payload += "=" * (-len(payload) % 4)
    try:
        return base64.b64decode(payload).decode("utf-8", "ignore")
    except Exception:
        return None


def search_candidates(query: str, limit: int = 14) -> list[str]:
    engines = [
        "https://www.bing.com/search?q=" + urllib.parse.quote(query) + "&setlang=zh-CN",
        "https://www.sogou.com/web?query=" + urllib.parse.quote(query),
    ]
    found: list[str] = []
    for engine in engines:
        try:
            _, raw = fetch(engine)
        except Exception:
            continue
        page = raw.decode("utf-8", "ignore")
        for match in re.finditer(r'href="(https?://[^"]+)"', page):
            raw_url = match.group(1).replace("&amp;", "&")
            target = decode_bing(raw_url) if "bing.com/ck/a" in raw_url else raw_url
            if not target:
                continue
            host = urllib.parse.urlparse(target).netloc
            if any(bad in host for bad in NOISE_HOSTS):
                continue
            if target not in found:
                found.append(target)
            if len(found) >= limit:
                return found
    return found


def main() -> int:
    specs = []
    for arg in sys.argv[1:]:
        if "|" in arg:
            code, name = arg.split("|", 1)
        else:
            code, name = arg, arg
        specs.append((code.strip(), name.strip()))
    report = []
    for code, name in specs:
        record: dict[str, object] = {"standard_code": code, "standard_name": name,
                                      "status": "pending", "candidates": [], "clauses": 0}
        slug = re.sub(r"[^0-9A-Za-z]+", "-", code).strip("-")
        query = f"{code} {name} 全文 条文"
        urls = search_candidates(query)
        urls.sort(key=lambda u: 0 if any(p in urllib.parse.urlparse(u).netloc for p in PREFERRED_HOSTS) else 1)
        best: tuple[int, str, bytes, list[dict[str, str]]] | None = None
        for url in urls[:10]:
            try:
                status, raw = fetch(url)
            except Exception as exc:  # noqa: BLE001
                record["candidates"].append({"url": url, "error": f"{type(exc).__name__}: {str(exc)[:70]}"})
                continue
            body = raw if raw[:4] == b"%PDF" else html_to_text(raw)
            clauses = parse_clauses(body)
            code_key = re.sub(r"[^0-9A-Za-z]", "", code).upper()
            page_key = re.sub(r"[^0-9A-Za-z]", "", body).upper()
            name_ok = (not name) or (name[:6] in body)
            relevant = code_key in page_key and name_ok
            record["candidates"].append({"url": url, "status": status, "bytes": len(raw),
                                         "is_pdf": raw[:4] == b"%PDF", "clauses_parsed": len(clauses),
                                         "relevant": relevant})
            if relevant and clauses and (best is None or len(clauses) > best[0]):
                best = (len(clauses), url, raw, clauses)
            if best and best[0] >= 40:
                break
        if best and best[0] < 20:
            record["status"] = "pending"
            record["note"] = "未找到含该标准全文的可靠来源，仅抓到零散条款，未采用。"
            best = None
        if best:
            count, url, raw, clauses = best
            sha = hashlib.sha256(raw).hexdigest()
            (OUT / f"{slug}.json").write_text(json.dumps({
                "standard_code": code, "standard_name": name, "source_url": url,
                "sha256": sha, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "clause_count": count, "clauses": clauses}, ensure_ascii=False, indent=2), encoding="utf-8")
            (OUT / f"{slug}.txt").write_text(
                html_to_text(raw) if raw[:4] != b"%PDF" else "", encoding="utf-8")
            record.update(status="ok", clauses=count, source_url=url, sha256=sha,
                          saved=str(OUT / f"{slug}.json"))
        report.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "candidates"}, ensure_ascii=False))
    (OUT / "fetch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
