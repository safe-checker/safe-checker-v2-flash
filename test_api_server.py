import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import api_server


class CropApiTests(unittest.TestCase):
    def setUp(self):
        self.auth = patch.object(api_server, "ACCESS_TOKEN", "")
        self.auth.start()
        self.client = TestClient(api_server.app)
        buffer = io.BytesIO()
        Image.new("RGB", (20, 10)).save(buffer, format="PNG")
        self.image = buffer.getvalue()

    def tearDown(self):
        self.client.close()
        self.auth.stop()

    def inspect(self, crop):
        return self.client.post("/v1/inspect", files={"image": ("test.png", self.image, "image/png")},
                                data={"detail_crop": crop})

    def test_crop_is_passed_to_one_checker_process(self):
        result = {"status": "success", "vision_calls": 1, "image_inputs": 2}
        with patch("api_server.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(result), stderr="",
        )) as run:
            response = self.inspect("100,200,700,800")
        self.assertEqual(response.status_code, 200)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--detail-crop") + 1], "100,200,700,800")
        self.assertEqual(run.call_count, 1)

    def test_invalid_crop_is_rejected_before_provider_call(self):
        with patch("api_server.subprocess.run") as run:
            response = self.inspect("0,0,1001,1000")
        self.assertEqual(response.status_code, 400)
        run.assert_not_called()

    def test_explicit_off_can_override_server_crop_setting(self):
        with patch("api_server.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout='{"status":"success"}', stderr="",
        )) as run:
            response = self.inspect("off")
        self.assertEqual(response.status_code, 200)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--detail-crop") + 1], "off")

    def test_model_deadline_remains_a_structured_timeout_at_api_boundary(self):
        failure = {"status": "error", "error": {"type": "timeout", "message": "budget exhausted"},
                   "elapsed_seconds": 7.6}
        with patch("api_server.subprocess.run", return_value=SimpleNamespace(
            returncode=1, stdout=json.dumps(failure), stderr="",
        )):
            response = self.inspect("auto")
        self.assertEqual(response.status_code, 504)
        # 对外错误契约是 FastAPI 的 {"detail": ...}，学长那边的视频端已经按这个接
        self.assertEqual(response.json(), {"detail": "budget exhausted"})


if __name__ == "__main__":
    unittest.main()


class ApiTransientRetryTests(unittest.TestCase):
    """A transient failure must be retried; a real answer must not be."""

    def _completed(self, payload: dict, returncode: int = 1):
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload), stderr="")

    def test_failure_type_is_read_from_the_structured_error(self) -> None:
        self.assertEqual(api_server._failure_type(self._completed(
            {"status": "error", "error": {"type": "timeout"}})), "timeout")
        self.assertEqual(api_server._failure_type(self._completed(
            {"status": "error", "error": {"type": "network_error"}})), "network_error")
        self.assertEqual(api_server._failure_type(self._completed({"status": "success"}, returncode=0)), "")
        self.assertEqual(api_server._failure_type(SimpleNamespace(
            returncode=1, stdout="not json", stderr="")), "unknown")

    def setUp(self):
        self.auth = patch.object(api_server, "ACCESS_TOKEN", "")
        self.auth.start()
        self.client = TestClient(api_server.app)
        buffer = io.BytesIO()
        Image.new("RGB", (20, 10)).save(buffer, format="PNG")
        self.image = buffer.getvalue()

    def tearDown(self):
        self.client.close()
        self.auth.stop()

    def test_transient_failure_is_retried_once(self) -> None:
        calls = {"n": 0}
        results = [
            self._completed({"status": "error", "error": {"type": "timeout"}}),
            self._completed({"status": "success", "issues": []}, returncode=0),
        ]

        def fake_run(*args, **kwargs):
            result = results[min(calls["n"], len(results) - 1)]
            calls["n"] += 1
            return result

        with patch.object(api_server.subprocess, "run", side_effect=fake_run), \
                patch.object(api_server, "TRANSIENT_RETRIES", 1), \
                patch.object(api_server, "TOTAL_TIMEOUT_SECONDS", 40.0):
            response = self.client.post("/v1/inspect", files={"image": ("t.png", self.image, "image/png")})
        self.assertEqual(calls["n"], 2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("status"), "success")

    def test_permanent_failure_is_not_retried(self) -> None:
        calls = {"n": 0}

        def fake_run(*args, **kwargs):
            calls["n"] += 1
            return self._completed({"status": "error", "error": {"type": "bad_request"}})

        with patch.object(api_server.subprocess, "run", side_effect=fake_run), \
                patch.object(api_server, "TRANSIENT_RETRIES", 1), \
                patch.object(api_server, "TOTAL_TIMEOUT_SECONDS", 40.0):
            response = self.client.post("/v1/inspect", files={"image": ("t.png", self.image, "image/png")})
        self.assertEqual(calls["n"], 1)
        self.assertEqual(response.status_code, 502)
