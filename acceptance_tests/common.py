#!/usr/bin/env python3
"""Shared helpers for the acceptance test scripts.

Only the public HTTP interface is used here. Nothing in this package
imports the detector internals, so the scripts can be handed to a third
party and executed against a running service without touching the source
tree.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Read the service token from the repository .env if it is there.

    The acceptance scripts are run from a checkout, so they should not need
    an extra export step just to authenticate against the service.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env()

BASE_URL = os.getenv("ACCEPTANCE_API_BASE", "http://127.0.0.1:8002")
API_KEY = os.getenv("API_ACCESS_TOKEN", "").strip()

# 10 construction scenes, matching the scene keys the detector reports.
SCENES: dict[str, str] = {
    "TEMPORARY_ELECTRICITY": "施工用电",
    "WORK_AT_HEIGHT": "高处作业",
    "FOUNDATION_PIT": "基坑工程",
    "LIFTING_OPERATIONS": "起重吊装",
    "TOWER_CRANE": "塔吊",
    "CONSTRUCTION_HOIST": "施工升降机",
    "SCAFFOLD_COUPLER_TYPE": "扣件式钢管脚手架",
    "SCAFFOLD_DISC_BUCKLE": "盘扣式脚手架",
    "SCAFFOLD_CANTILEVER": "悬挑式脚手架",
    "SCAFFOLD_ATTACHED_LIFTING": "附着升降脚手架",
}
# 脚手架四个型式在图库里并成一类，按型式细分需要额外标注。
SCAFFOLD_ALIASES = {
    "SCAFFOLD_FAMILY": "SCAFFOLD_COUPLER_TYPE",
}

# 考核指标 P7：风险预警延迟不大于 8 秒。
LATENCY_THRESHOLD = float(os.getenv("ACCEPTANCE_LATENCY_THRESHOLD", "8.0"))


def scene_name(code: str) -> str:
    code = SCAFFOLD_ALIASES.get(code, code)
    return SCENES.get(code, code)


class Inspector:
    """Thin client for POST /v1/inspect."""

    def __init__(self, base_url: str = BASE_URL, api_key: str = API_KEY,
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Inspector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def health(self) -> tuple[int, Any, float]:
        started = time.perf_counter()
        response = self._client.get(f"{self.base_url}/health")
        elapsed = time.perf_counter() - started
        return response.status_code, _safe_json(response), elapsed

    def inspect(self, image_path: str | Path, *, scene: str = "", prompt: str = "",
                detail_crop: str = "") -> tuple[int, Any, float]:
        """Return (status_code, payload, wall_seconds).

        wall_seconds is measured around the request only. It is the number
        the P7 metric is judged on, so it must not include client-side
        bookkeeping.
        """
        with open(image_path, "rb") as handle:
            files = {"image": (Path(image_path).name, handle, "application/octet-stream")}
            data: dict[str, str] = {}
            if scene:
                data["scene"] = scene
            if prompt:
                data["prompt"] = prompt
            if detail_crop:
                data["detail_crop"] = detail_crop
            started = time.perf_counter()
            response = self._client.post(
                f"{self.base_url}/v1/inspect", files=files, data=data, headers=self.headers()
            )
            elapsed = time.perf_counter() - started
        return response.status_code, _safe_json(response), elapsed

    def post_raw(self, path: str, *, files: dict | None = None,
                 data: dict | None = None, headers: dict | None = None,
                 content: bytes | None = None) -> tuple[int, Any, float]:
        """POST with an arbitrary body. Returns (status_code, payload, wall_seconds)."""
        started = time.perf_counter()
        response = self._client.post(
            f"{self.base_url}{path}", files=files, data=data or {},
            headers=headers or self.headers(), content=content,
        )
        elapsed = time.perf_counter() - started
        return response.status_code, _safe_json(response), elapsed


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"detail": response.text[:2000]}


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values: list[float], ratio: float) -> float:
    """Nearest-rank percentile. Small samples stay honest this way."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(ratio * len(ordered) + 0.5)) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
        "median": round(percentile(ordered, 0.50), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
