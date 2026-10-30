import contextlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_claude_smoke as smoke

KEY = "test-private-credential-123456789"


def message(model, **changes):
    obj = {"type": "message", "role": "assistant", "model": model,
           "id": "msg_fixture", "usage": {"input_tokens": 90, "output_tokens": 180},
           "content": [{"type": "text", "text": "A small book table serves the community."}],
           "stop_reason": "end_turn"}
    obj.update(changes)
    return obj


class Response(io.BytesIO):
    def __init__(self, obj=None, status=200, raw=None, headers=None):
        super().__init__(raw if raw is not None else json.dumps(obj).encode())
        self.status = status
        self.headers = headers or {"request-id": "req_fixture"}


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "smoke"
        self.calls = []

    def sender(self, req, timeout):
        self.calls.append(req)
        body = json.loads(req.data)
        return Response(message(body["model"]))

    def execute(self, opener=None, **kwargs):
        return smoke.run(True, self.run_dir, environ={"ANTHROPIC_API_KEY": KEY},
                         opener=opener or self.sender, **kwargs)

    def test_default_preview_never_requests_or_reads_key(self):
        with patch.object(smoke, "acquire_key", side_effect=AssertionError):
            code, result = smoke.run(False, self.run_dir, opener=self.sender)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "preview_only_no_requests")
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(self.calls, [])

    def test_three_requests_fixed_models_endpoint_and_budget(self):
        code, summary = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(summary["status"], "three_generations_recorded")
        for req, model in zip(self.calls, smoke.MODELS):
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
            self.assertEqual(req.get_header("X-api-key"), KEY)
            self.assertEqual(json.loads(req.data), smoke.request_body(model))
            self.assertEqual(json.loads(req.data)["max_tokens"], 512)
            self.assertNotIn("temperature", json.loads(req.data))
            self.assertNotIn("tools", json.loads(req.data))
        self.assertLess(summary["estimated_standard_token_cost_usd"], .10)
        self.assertAlmostEqual(smoke.plan()["estimated_max_cost_usd"], .0339456)

    def test_sonnet55_thinking_is_between_tools_not_disabled(self):
        self.assertEqual(smoke.request_body("claude-sonnet-5-5")["thinking"], {"type": "between_tools"})
        for model in smoke.MODELS:
            self.assertEqual(smoke.request_body(model)["output_config"], {"effort": "high"})
            if model != "claude-sonnet-5-5":
                self.assertEqual(smoke.request_body(model)["thinking"], {"type": "disabled"})

    def test_existing_directory_refused_before_key_or_network(self):
        self.run_dir.mkdir()
        with patch.object(smoke, "acquire_key", side_effect=AssertionError):
            with self.assertRaises(smoke.CheckError):
                self.execute()
        self.assertEqual(self.calls, [])

    def test_existing_archive_refused_before_key_or_network(self):
        self.run_dir.with_suffix(".zip").write_bytes(b"old")
        with patch.object(smoke, "acquire_key", side_effect=AssertionError):
            with self.assertRaises(smoke.CheckError):
                self.execute()
        self.assertEqual(self.run_dir.with_suffix(".zip").read_bytes(), b"old")

    def test_missing_key_no_network_or_run_reservation(self):
        with self.assertRaises(smoke.CheckError):
            smoke.run(True, self.run_dir, environ={}, stdin=io.StringIO(), opener=self.sender)
        self.assertFalse(self.run_dir.exists())
        self.assertEqual(self.calls, [])

    def test_request_ledger_precedes_post(self):
        def send(req, timeout):
            index = len(self.calls) + 1
            attempt = json.loads((self.run_dir / ("attempt-%02d.json" % index)).read_text())
            summary = json.loads((self.run_dir / "summary.json").read_text())
            self.assertEqual(attempt["status"], "request_started")
            self.assertEqual(summary["generation_requests_attempted"], index)
            return self.sender(req, timeout)
        self.execute(send)

    def test_transport_error_stops_first_no_retry_and_keeps_bundle(self):
        def send(req, timeout):
            self.calls.append(req)
            raise urllib.error.URLError(KEY)
        code, summary = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(summary["status"], "stopped_on_first_failure")
        self.assertTrue(self.run_dir.with_suffix(".zip").exists())
        self.assertNotIn(KEY, (self.run_dir / "attempt-01.json").read_text())

    def test_http_error_body_and_request_id_key_are_redacted(self):
        def send(req, timeout):
            self.calls.append(req)
            raise urllib.error.HTTPError(req.full_url, 401, KEY,
                {"request-id": KEY}, io.BytesIO(json.dumps({"error": {"message": KEY}}).encode()))
        code, _ = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(KEY, (self.run_dir / "attempt-01.json").read_text())
        self.assertIn("REDACTED_CREDENTIAL", (self.run_dir / "attempt-01.json").read_text())

    def test_escaped_json_credential_echo_redacted_before_persistence(self):
        def send(req, timeout):
            self.calls.append(req)
            text = message(json.loads(req.data)["model"])
            text["content"][0]["text"] = KEY
            raw = json.dumps(text).replace(KEY, "".join("\\u%04x" % ord(c) for c in KEY)).encode()
            return Response(raw=raw)
        code, summary = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(summary["results"][0]["status"], "credential_echo_redacted_stop")
        with zipfile.ZipFile(self.run_dir.with_suffix(".zip")) as z:
            for name in z.namelist():
                self.assertNotIn(KEY.encode(), z.read(name))

    def test_double_encoded_credential_echo_is_redacted(self):
        escaped = "".join("\\u%04x" % ord(c) for c in KEY)
        def send(req, timeout):
            self.calls.append(req)
            return Response(message(json.loads(req.data)["model"],
                                    content=[{"type": "text", "text": escaped}]))
        code, _ = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(escaped, (self.run_dir / "attempt-01.json").read_text())
        self.assertIn("REDACTED_CREDENTIAL", (self.run_dir / "attempt-01.json").read_text())

    def test_large_integer_json_failure_has_no_traceback_or_next_post(self):
        def send(req, timeout):
            self.calls.append(req)
            return Response(raw=b'{"value":' + b'9' * 5000 + b'}')
        code, _ = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_read_has_no_raw_exception_output(self):
        class Broken(Response):
            def read(self, size=-1):
                raise http.client.IncompleteRead(KEY.encode(), 50)
        def send(req, timeout):
            self.calls.append(req)
            return Broken({})
        code, _ = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(KEY, (self.run_dir / "attempt-01.json").read_text())

    def test_invalid_model_stops_remaining_requests(self):
        def send(req, timeout):
            self.calls.append(req)
            return Response(message("unrequested-model"))
        code, result = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(result["results"][0]["status"], "invalid_message_identity")
        self.assertEqual(len(self.calls), 1)

    def test_invalid_usage_stops_remaining_requests(self):
        def send(req, timeout):
            self.calls.append(req)
            return Response(message(json.loads(req.data)["model"], usage={"input_tokens": 4097, "output_tokens": 20}))
        code, result = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(result["results"][0]["status"], "token_planning_allowance_exceeded")
        self.assertEqual(len(self.calls), 1)

    def test_output_limit_retained_without_regeneration(self):
        def send(req, timeout):
            return Response(message(json.loads(req.data)["model"], stop_reason="max_tokens"))
        code, _ = self.execute(send)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads((self.run_dir / "attempt-01.json").read_text())["result"]["hit_output_limit"])

    def test_persistence_failure_prevents_first_request(self):
        with patch.object(smoke, "save_owned", side_effect=OSError(KEY)):
            with self.assertRaises(OSError):
                self.execute()
        self.assertEqual(self.calls, [])

    def test_persistence_failure_after_paid_post_prevents_next_post(self):
        original_save = smoke.save_owned
        def save(path, record):
            if path.name == "attempt-01.json":
                raise OSError(KEY)
            return original_save(path, record)
        with patch.object(smoke, "save_owned", side_effect=save):
            code, summary = self.execute()
        self.assertEqual(code, 2)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(summary["status"], "persistence_failed_no_retry")
        self.assertFalse(summary["cost_estimate_covers_all_attempts"])
        self.assertTrue(self.run_dir.with_suffix(".zip").exists())

    def test_bundle_has_only_declared_files(self):
        def send(req, timeout):
            (self.run_dir / ".env").write_text(KEY)
            return self.sender(req, timeout)
        self.execute(send)
        with zipfile.ZipFile(self.run_dir.with_suffix(".zip")) as z:
            self.assertEqual(set(z.namelist()), {"plan.json", "summary.json", "attempt-01.json", "attempt-02.json", "attempt-03.json"})

    def test_response_size_bounded(self):
        def send(req, timeout):
            self.calls.append(req)
            return Response(raw=b"x" * (smoke.MAX_RESPONSE_BYTES + 2))
        code, _ = self.execute(send)
        self.assertEqual(code, 3)
        self.assertEqual(len(self.calls), 1)

    def test_no_redirects(self):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages")
        self.assertIsNone(smoke.NoRedirect().redirect_request(req, None, 302, "", {}, "https://example.invalid"))

    def test_invalid_cli_does_not_echo_secret(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            smoke.main([KEY])
        self.assertNotIn(KEY, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
