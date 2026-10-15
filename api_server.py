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
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
CHECK_SCRIPT = BASE_DIR / "safety_check.py"
load_dotenv(BASE_DIR / ".env")

MAX_UPLOAD_BYTES = int(os.getenv("API_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
PROCESS_TIMEOUT_SECONDS = float(os.getenv("API_PROCESS_TIMEOUT_SECONDS", "8"))
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("API_MAX_CONCURRENT_REQUESTS", "2")))
ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN", "").strip()
REQUEST_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Construction Safety Image Inspection API",
    version="1.0.0",
    description="Upload a construction-site image and receive a structured single-call DeepSeek Flash safety inspection with original-image bounding boxes.",
)


def _process_failure(completed: subprocess.CompletedProcess[str], result: Any) -> HTTPException:
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/inspect", response_model=dict)
@app.post("/api/v1/inspect", include_in_schema=False)
async def inspect_image(
    image: UploadFile = File(..., description="A construction-site image file."),
    scene: str = Form("", description="Optional scene, for example: 起重吊装 or 高处作业."),
    prompt: str = Form("", description="Optional additional inspection instruction."),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> JSONResponse:
    if ACCESS_TOKEN and not hmac.compare_digest(x_api_key or "", ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")

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

            try:
                completed = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=PROCESS_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.warning("Inspection subprocess exceeded API_PROCESS_TIMEOUT_SECONDS=%s", PROCESS_TIMEOUT_SECONDS)
                raise HTTPException(
                    status_code=504,
                    detail=f"Inspection exceeded the {PROCESS_TIMEOUT_SECONDS:g}-second API processing limit.",
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
