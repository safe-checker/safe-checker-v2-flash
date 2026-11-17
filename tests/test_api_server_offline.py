"""HTTP error-contract regressions with no real child processes or model calls."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncio
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
import warnings
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import api_server
from safety_check import analyze_image_quality


PILLOW_WARNING = "safety_check.py:174: DeprecationWarning: Image.Image.getdata is deprecated\n  pixels = list(gray.getdata())"


class ApiResultCacheTests(unittest.TestCase):
    """The idempotency cache: the same picture must give the same answer."""

    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (128, 128, 128)).save(buffer, format="PNG")
        self.image = buffer.getvalue()
        api_server._RESULT_CACHE.clear()
        self.addCleanup(api_server._RESULT_CACHE.clear)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.addCleanup(loop.close)
        for name, value in (("ACCESS_TOKEN", ""), ("REQUEST_SLOTS", asyncio.Semaphore(2)),
                            ("CACHE_ENABLED", True)):
            patcher = patch.object(api_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        client_patcher = TestClient(api_server.app)
        client_patcher.__enter__()
        self.addCleanup(client_patcher.__exit__, None, None, None)
        self.client = client_patcher
        self.payload = {"status": "success", "issues": [], "issue_count": 0}

    def _post(self, content, scene=""):
        with patch.object(api_server.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0, json.dumps(self.payload), "")) as run:
            response = self.client.post(
                "/v1/inspect", files={"image": ("frame.png", content, "image/png")},
                data={"scene": scene} if scene else None)
        return response, run

    def test_same_picture_is_served_from_cache(self):
        first, run_first = self._post(self.image)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers.get("X-Cache"), "MISS")
        run_first.assert_called_once()

        second, run_second = self._post(self.image)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers.get("X-Cache"), "HIT")
        run_second.assert_not_called()
        self.assertEqual(first.json(), second.json())

    def test_different_scene_is_not_cached(self):
        self._post(self.image, scene="高处作业")
        response, run = self._post(self.image, scene="高处作业")
        self.assertEqual(response.headers.get("X-Cache"), "HIT")
        run.assert_not_called()

        response, run = self._post(self.image, scene="起重吊装")
        self.assertEqual(response.headers.get("X-Cache"), "MISS")
        run.assert_called_once()

    def test_different_picture_is_not_cached(self):
        self._post(self.image)
        other = io.BytesIO()
        Image.new("RGB", (16, 12), (10, 20, 30)).save(other, format="PNG")
        response, run = self._post(other.getvalue())
        self.assertEqual(response.headers.get("X-Cache"), "MISS")
        run.assert_called_once()

    def test_partial_branch_result_is_not_cached(self):
        self.payload = {
            "status": "success",
            "completion_status": "partial",
            "issues": [],
            "issue_count": 0,
        }
        first, run_first = self._post(self.image)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers.get("X-Cache"), "MISS")
        run_first.assert_called_once()

        second, run_second = self._post(self.image)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.headers.get("X-Cache"), "MISS")
        run_second.assert_called_once()


class ApiErrorTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (128, 128, 128)).save(buffer, format="PNG")
        self.image = buffer.getvalue()
        # 本仓库的 .venv 是 Python 3.9，没有 unittest.TestCase.enterContext（3.11+），
        # 这里用 addCleanup 保持同样的语义。
        # 别的测试模块跑过 asyncio.run() 之后，全局事件循环会被清空。
        # Python 3.9 的 Semaphore 与 TestClient 都依赖当前循环，所以先建循环再建它们。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.addCleanup(loop.close)
        # 错误契约测试关心的是失败路径，幂等缓存会跨用例复用结果，这里显式关掉。
        for name, value in (("ACCESS_TOKEN", ""), ("REQUEST_SLOTS", asyncio.Semaphore(2)),
                            ("CACHE_ENABLED", False)):
            patcher = patch.object(api_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        client_patcher = TestClient(api_server.app)
        client_patcher.__enter__()
        self.addCleanup(client_patcher.__exit__, None, None, None)
        self.client = client_patcher

    def inspect(self, completed, retries: int = 0):
        """One inspection call. `retries` defaults to 0 so the error contract is
        tested without the transient-retry layer in the way."""
        with patch.object(api_server, "TRANSIENT_RETRIES", retries), \
                patch.object(api_server.subprocess, "run", return_value=completed) as run:
            response = self.client.post("/v1/inspect", files={"image": ("scene.png", self.image, "image/png")})
        if retries == 0:
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


class TransientRetryTests(unittest.TestCase):
    """A timeout is worth another attempt; a real answer is not."""

    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 12), (128, 128, 128)).save(buffer, format="PNG")
        self.image = buffer.getvalue()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.addCleanup(loop.close)
        # 错误契约测试关心的是失败路径，幂等缓存会跨用例复用结果，这里显式关掉。
        for name, value in (("ACCESS_TOKEN", ""), ("REQUEST_SLOTS", asyncio.Semaphore(2)),
                            ("CACHE_ENABLED", False)):
            patcher = patch.object(api_server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_timeout_is_retried_once_and_still_reports_504(self):
        failure = {"status": "error", "error": {"type": "timeout", "message": "budget exhausted"}}
        completed = subprocess.CompletedProcess([], 1, json.dumps(failure), "")
        with patch.object(api_server, "TRANSIENT_RETRIES", 1), \
                patch.object(api_server.subprocess, "run", return_value=completed) as run:
            with TestClient(api_server.app) as client:
                response = client.post(
                    "/v1/inspect", files={"image": ("scene.png", self.image, "image/png")})
        self.assertEqual(run.call_count, 2)
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json(), {"detail": "budget exhausted"})

    def test_a_real_process_error_is_not_retried(self):
        failure = {"status": "error", "error": {"type": "processing_error", "message": "bad report"}}
        completed = subprocess.CompletedProcess([], 1, json.dumps(failure), "")
        with patch.object(api_server, "TRANSIENT_RETRIES", 1), \
                patch.object(api_server.subprocess, "run", return_value=completed) as run:
            with TestClient(api_server.app) as client:
                response = client.post(
                    "/v1/inspect", files={"image": ("scene.png", self.image, "image/png")})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(response.status_code, 502)


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


class SceneChecklistTests(unittest.TestCase):
    """The scene value decides what the model is told to look for."""

    def setUp(self):
        # 这些环境变量决定提示词里的清单内容，用完必须还原，
        # 否则会污染同一进程里后面跑的用例。
        self._saved = {
            name: os.environ.get(name)
            for name in ("SCENE_CHECKLIST_SCOPE", "CHECKLIST_SUBJECT_GROUP",
                         "PART_CHECKLIST_IN_FAMILY")
        }
        os.environ["SCENE_CHECKLIST_SCOPE"] = "family_parts"
        os.environ["CHECKLIST_SUBJECT_GROUP"] = "0/4"
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _checklist(self, scene_key):
        import safety_check as checker
        return checker.build_scene_checklist_section(scene_key)

    def test_comprehensive_scene_covers_more_than_the_person_subject(self):
        """施工现场综合安全 must not collapse to the single "图片中人员" subject.

        GENERAL_SITE only owns one subject of its own, so using it directly
        made the comprehensive scene narrower than leaving the scene unset.
        """
        text = self._checklist("GENERAL_SITE")
        self.assertIn("施工现场综合安全", text.split("\n")[0])
        self.assertGreater(text.count("主体["), 5)

    def test_electricity_scene_carries_part_level_details(self):
        """The part-level checklist has to reach the family scenes.

        Without it the model is never told to look for a loose fuse, a
        missing fire extinguisher or a bent scaffold pole. The rows are split
        across the branches, so all four have to be checked together.
        """
        import safety_check as checker
        combined = ""
        for group in ("0/4", "1/4", "2/4", "3/4"):
            os.environ["CHECKLIST_SUBJECT_GROUP"] = group
            combined += checker.build_scene_checklist_section("TEMPORARY_ELECTRICITY")
        self.assertIn("部件细项", combined)
        for keyword in ("熔断器", "灭火器", "插座", "接地"):
            self.assertIn(keyword, combined)

    def test_part_rows_can_be_switched_off_for_comparison(self):
        import safety_check as checker
        previous = os.environ.get("PART_CHECKLIST_IN_FAMILY")
        try:
            os.environ["PART_CHECKLIST_IN_FAMILY"] = "0"
            self.assertNotIn("部件细项", self._checklist("TEMPORARY_ELECTRICITY"))
            os.environ["PART_CHECKLIST_IN_FAMILY"] = "1"
            self.assertIn("部件细项", self._checklist("TEMPORARY_ELECTRICITY"))
        finally:
            if previous is None:
                os.environ.pop("PART_CHECKLIST_IN_FAMILY", None)
            else:
                os.environ["PART_CHECKLIST_IN_FAMILY"] = previous
