"""HTTP error-contract regressions with no real child processes or model calls."""
from __future__ import annotations

import asyncio
import io
import json
import signal
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import api_server
from safety_check import analyze_image_quality


PILLOW_WARNING = "safety_check.py:174: DeprecationWarning: Image.Image.getdata is deprecated\n  pixels = list(gray.getdata())"


class ApiErrorTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (128, 128, 128)).save(buffer, format="PNG")
        self.image = buffer.getvalue()
        self.stack = self.enterContext(patch.object(api_server, "ACCESS_TOKEN", ""))
        self.enterContext(patch.object(api_server, "REQUEST_SLOTS", asyncio.Semaphore(2)))
        self.client = self.enterContext(TestClient(api_server.app))

    def inspect(self, completed):
        with patch.object(api_server.subprocess, "run", return_value=completed) as run:
            response = self.client.post("/v1/inspect", files={"image": ("scene.png", self.image, "image/png")})
        run.assert_called_once()
        return response

    def test_json_error_is_not_hidden_by_stderr_warning(self):
        for error_type, message, code, expected in [
            ("timeout", "DeepSeek Flash request exceeded the time budget (8.0s total).", 1, 504),
            ("missing_api_key", "Please set DEEPSEEK_API_KEY in .env first.", 2, 502),
            ("processing_error", "The model returned an empty report.", 1, 502),
        ]:
            with self.subTest(error_type=error_type):
                result = {"status": "error", "error": {"type": error_type, "message": message}}
                response = self.inspect(subprocess.CompletedProcess([], code, json.dumps(result), PILLOW_WARNING))
                self.assertEqual(response.status_code, expected)
                self.assertEqual(response.json(), {"detail": message})

    def test_signal_termination_is_explicit_even_when_stderr_contains_only_a_warning(self):
        response = self.inspect(subprocess.CompletedProcess([], -signal.SIGKILL, "", PILLOW_WARNING))
        self.assertEqual(response.status_code, 502)
        self.assertIn("SIGKILL", response.json()["detail"])
        self.assertNotIn("DeprecationWarning", response.json()["detail"])

    def test_unstructured_failures_preserve_diagnostic_and_exit_code(self):
        response = self.inspect(subprocess.CompletedProcess([], 3, "", "fixture process failure"))
        self.assertEqual(response.status_code, 502)
        self.assertIn("exit code 3", response.json()["detail"])
        self.assertIn("fixture process failure", response.json()["detail"])

    def test_success_with_stderr_warning_stays_successful(self):
        payload = {"status": "success", "issues": [], "issue_count": 0}
        response = self.inspect(subprocess.CompletedProcess([], 0, json.dumps(payload), PILLOW_WARNING))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)

    def test_error_payload_does_not_become_a_success_on_zero_exit(self):
        payload = {"status": "error", "error": {"type": "processing_error", "message": "fixture error"}}
        response = self.inspect(subprocess.CompletedProcess([], 0, json.dumps(payload), ""))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "fixture error")

    def test_invalid_or_non_object_success_output_is_rejected(self):
        for output in ["not JSON", "[]", "null"]:
            with self.subTest(output=output):
                response = self.inspect(subprocess.CompletedProcess([], 0, output, ""))
                self.assertEqual(response.status_code, 502)
                self.assertIn("invalid JSON", response.json()["detail"])

    def test_api_processing_deadline_remains_a_gateway_timeout(self):
        with patch.object(api_server.subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 8)):
            response = self.client.post("/v1/inspect", files={"image": ("scene.png", self.image, "image/png")})
        self.assertEqual(response.status_code, 504)
        self.assertIn("API processing limit", response.json()["detail"])
        response = self.inspect(subprocess.CompletedProcess([], 0, '{"status":"success","issues":[]}', ""))
        self.assertEqual(response.status_code, 200)


class ImageQualityCompatibilityTests(unittest.TestCase):
    def test_quality_analysis_preserves_pixels_without_pillow_deprecation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quality.png"
            image = Image.new("L", (20, 10), 128)
            image.paste(255, (0, 0, 10, 10))
            image.save(path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                quality = analyze_image_quality(path)
        self.assertEqual((quality["width"], quality["height"]), (20, 10))
        self.assertEqual(quality["overexposed_ratio"], 0.5)
        self.assertEqual(quality["underexposed_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
