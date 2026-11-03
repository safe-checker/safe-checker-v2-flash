#!/usr/bin/env python3
"""HTTP API wrapper for the construction safety image checker."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import os
import signal
import subprocess
import time
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# 仓库根目录（.env 在那里），代码目录（safety_check.py 在这里）
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
CHECK_SCRIPT = BASE_DIR / "safety_check.py"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from safety_check import parse_detail_crop  # noqa: E402

load_dotenv(REPO_DIR / ".env")

MAX_UPLOAD_BYTES = int(os.getenv("API_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
# 检测器自己的检测预算；进程硬杀要比它宽松，才能拿到结构化超时错误
INSPECTION_BUDGET_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "8"))
# 一次请求内部并发几路检查。1 = 不并发；2 和 4 的准确率实测相同，2 更稳。
BRANCHES = int(os.getenv("API_BRANCHES", "2"))
PROCESS_TIMEOUT_SECONDS = float(
    os.getenv("API_PROCESS_TIMEOUT_SECONDS", str(INSPECTION_BUDGET_SECONDS + 1.5))
)
# 整体调用允许的最长时间预算；必须够一次完整重试，重试才会真的发生
TOTAL_TIMEOUT_SECONDS = float(
    os.getenv("API_TOTAL_TIMEOUT_SECONDS", str(INSPECTION_BUDGET_SECONDS * 2 + 1.5))
)
# 瞬时故障（超时、网络错误）时额外重试的次数
TRANSIENT_RETRIES = max(0, int(os.getenv("API_TRANSIENT_RETRIES", "1")))
RETRYABLE_ERRORS = {"timeout", "network_error"}
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("API_MAX_CONCURRENT_REQUESTS", "2")))
ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()
REQUEST_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

def _process_failure(completed: subprocess.CompletedProcess[str], result: Any) -> HTTPException:
    """Map a failed inspection to the HTTP contract the video client expects.

    The detection script writes a structured error to stdout; warnings from
    Pillow and friends go to stderr and must never hide the real reason.
    A process killed by a signal is reported by signal name, because a bare
    exit code does not tell the operator what happened.
    """
    error = result.get("error") if isinstance(result, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            error_type = error.get("type")
            logger.warning("Inspection failed: returncode=%s error_type=%s", completed.returncode, error_type)
            return HTTPException(
                status_code=504 if error_type == "timeout" else 502,
                detail=message.strip()[-2000:],
            )
    if completed.returncode < 0:
        number = -completed.returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        message = f"Inspection process was terminated by signal {name} ({number}). Check the service and system logs."
    else:
        diagnostic = completed.stderr.strip() or completed.stdout.strip() or "No error details were returned."
        message = f"Inspection process failed (exit code {completed.returncode}): {diagnostic[-1800:]}"
    logger.warning("Inspection process failed: returncode=%s", completed.returncode)
    return HTTPException(status_code=502, detail=message)


def _failure_type(completed: subprocess.CompletedProcess) -> str:
    """Return the structured error type of a failed inspection, if any."""
    if completed.returncode == 0:
        return ""
    try:
        failure = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        return "unknown"
    if isinstance(failure, dict) and failure.get("status") == "error":
        error = failure.get("error")
        if isinstance(error, dict):
            return str(error.get("type") or "unknown")
    return "unknown"


app = FastAPI(
    title="Construction Safety Image Inspection API",
    version="1.0.0",
    description="Upload a construction-site image and receive a structured single-call DeepSeek Flash safety inspection with original-image bounding boxes.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/inspect", response_model=dict)
@app.post("/api/v1/inspect", include_in_schema=False)
async def inspect_image(
    image: UploadFile = File(..., description="A construction-site image file."),
    scene: str = Form("", description="Optional scene, for example: 起重吊装 or 高处作业."),
    prompt: str = Form("", description="Optional additional inspection instruction."),
    detail_crop: str = Form(
        "",
        description="Optional second image crop: auto or x1,y1,x2,y2 on a 0..1000 grid.",
    ),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> JSONResponse:
    if ACCESS_TOKEN and not hmac.compare_digest(x_api_key or "", ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")

    if detail_crop.strip().lower() != "auto":
        try:
            parse_detail_crop(detail_crop, (1000, 1000))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
        )

    try:
        with Image.open(io.BytesIO(content)) as opened:
            image_format = (opened.format or "").upper()
            opened.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=415, detail="The uploaded file is not a supported image.")

    suffix = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "BMP": ".bmp",
    }.get(image_format)
    if suffix is None:
        raise HTTPException(status_code=415, detail=f"Unsupported image format: {image_format or 'unknown'}.")

    try:
        await asyncio.wait_for(REQUEST_SLOTS.acquire(), timeout=0.05)
    except TimeoutError:
        raise HTTPException(status_code=429, detail="Inspection capacity is busy; retry shortly.")

    try:
        with tempfile.NamedTemporaryFile(prefix="safety-inspect-", suffix=suffix) as temp_image:
            temp_image.write(content)
            temp_image.flush()
            command = [
                sys.executable,
                str(CHECK_SCRIPT),
                temp_image.name,
                "--no-save",
                "--json-output",
            ]
            if scene.strip():
                command.extend(["--scene", scene.strip()])
            if prompt.strip():
                command.extend(["--prompt", prompt.strip()])
            if detail_crop.strip():
                command.extend(["--detail-crop", detail_crop.strip()])
            # 并发分支数：2 路和 4 路准确率相同（同图配对 64% vs 64%），
            # 2 路的上游压力只有一半，所以默认 2；视频端要更稳可以配 4。
            if BRANCHES > 1:
                command.extend(["--branches", str(BRANCHES)])

            child_env = {
                **os.environ,
                "REQUEST_TIMEOUT_SECONDS": str(INSPECTION_BUDGET_SECONDS),
            }
            total_deadline = time.monotonic() + TOTAL_TIMEOUT_SECONDS
            completed = None
            for attempt in range(TRANSIENT_RETRIES + 1):
                remaining_total = total_deadline - time.monotonic()
                if remaining_total <= 0:
                    break
                try:
                    completed = await asyncio.to_thread(
                        subprocess.run,
                        command,
                        cwd=BASE_DIR,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=min(PROCESS_TIMEOUT_SECONDS, remaining_total),
                        check=False,
                        env=child_env,
                    )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Inspection subprocess exceeded API_PROCESS_TIMEOUT_SECONDS=%s", PROCESS_TIMEOUT_SECONDS
                    )
                    completed = None
                    break
                failure_type = _failure_type(completed)
                if failure_type not in RETRYABLE_ERRORS:
                    break
                if attempt >= TRANSIENT_RETRIES:
                    break
                if total_deadline - time.monotonic() < INSPECTION_BUDGET_SECONDS:
                    break
            if completed is None:
                raise HTTPException(
                    status_code=504,
                    detail=f"Inspection exceeded the {TOTAL_TIMEOUT_SECONDS:g}-second API processing limit.",
                )

        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = None
        if completed.returncode != 0 or (isinstance(result, dict) and result.get("status") == "error"):
            raise _process_failure(completed, result)
        if not isinstance(result, dict):
            raise HTTPException(status_code=502, detail="The inspection process returned invalid JSON.")
        return JSONResponse(result)
    finally:
        REQUEST_SLOTS.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8002")),
        reload=False,
    )
