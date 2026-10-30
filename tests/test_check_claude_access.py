"""Offline tests; never use real credentials or network access."""
import contextlib
import importlib.util
import io
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import warnings

SPEC = importlib.util.spec_from_file_location("check_claude_access", Path(__file__).resolve().parents[1] / "scripts/check_claude_access.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
KEY = "sk-ant-offline-fixture-0123456789"


class Response:
    def __init__(self, obj=None, status=200, raw=None):
        self.status = status
        self.raw = json.dumps(obj).encode() if raw is None else raw
        self.read_limit = None
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, n):
        self.read_limit = n
        return self.raw[:n]


def page(ids, more=False, **extra):
    return {"data": [{"id": x, "type": "model"} for x in ids], "has_more": more,
            "last_id": ids[-1] if ids else None, **extra}


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "report.json"
        self.calls = []
    def sender(self, *replies):
        iterator = iter(replies)
        def send(req, timeout):
            self.calls.append(req)
            self.assertEqual(timeout, 30)
            reply = next(iterator)
            if isinstance(reply, BaseException):
                raise reply
            return Response(reply)
        return send
    def run_check(self, *replies):
        return check.run(self.output, environ={"ANTHROPIC_API_KEY": KEY}, opener=self.sender(*replies))
    def test_two_pages_only_get_with_correct_headers_and_bound(self):
        code, record = self.run_check(page(["claude-a"], True), page(["claude-b"]))
        self.assertEqual(code, 0)
        self.assertEqual(record["model_ids"], ["claude-a", "claude-b"])
        self.assertTrue(all(req.method == "GET" and req.data is None for req in self.calls))
        self.assertEqual(self.calls[0].full_url, check.API_ROOT + "/v1/models?limit=100")
        self.assertTrue(self.calls[1].full_url.endswith("after_id=claude-a"))
        self.assertEqual(dict(self.calls[0].header_items())["X-api-key"], KEY)
        self.assertEqual(dict(self.calls[0].header_items())["Anthropic-version"], check.API_VERSION)
        self.assertFalse(record["credit_balance_checked"])
        self.assertFalse(record["inference_verified"])
        self.assertEqual(record["generation_requests_sent"], 0)
        self.assertNotIn(KEY, self.output.read_text())
    def test_ledger_is_persisted_before_request(self):
        def send(req, timeout):
            record = json.loads(self.output.read_text())
            self.assertEqual(record["pages"][0]["status"], "request_started")
            return Response(page([]))
        code, _ = check.run(self.output, environ={"ANTHROPIC_API_KEY": KEY}, opener=send)
        self.assertEqual(code, 0)
    def test_existing_file_is_not_overwritten_or_requested(self):
        self.output.write_text("old")
        with self.assertRaises(FileExistsError):
            self.run_check(page([]))
        self.assertEqual(self.output.read_text(), "old")
        self.assertEqual(self.calls, [])
    def test_repeated_results_stop_without_retry(self):
        code, record = self.run_check(page(["claude-a"], True), page(["claude-a"], True))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "repeated_pagination_results")
        self.assertEqual(len(self.calls), 2)
    def test_invalid_cursor_not_requested_or_saved(self):
        code, record = self.run_check(page(["claude-a"], True, last_id=KEY))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "invalid_pagination_cursor")
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn(KEY, self.output.read_text())
    def test_secret_echo_in_non_whitelisted_fields_is_discarded(self):
        obj = page(["claude-a"], display_name=KEY, error={"message": KEY}, last_id=KEY)
        obj["data"][0]["display_name"] = KEY
        code, record = self.run_check(obj)
        self.assertEqual(code, 0)
        self.assertNotIn(KEY, json.dumps(record))
    def test_secret_echo_in_model_id_is_rejected(self):
        code, record = self.run_check(page(["claude-" + KEY]))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "invalid_or_sensitive_model_id")
        self.assertNotIn(KEY, self.output.read_text())
    def test_http_error_body_and_message_are_never_retained(self):
        error = urllib.error.HTTPError("https://api.anthropic.com", 401, KEY, {"request-id": KEY}, io.BytesIO(KEY.encode()))
        code, record = self.run_check(error)
        self.assertEqual(code, 3)
        self.assertEqual(record["pages"][0]["http_status"], 401)
        self.assertNotIn(KEY, self.output.read_text())
        self.assertEqual(len(self.calls), 1)
    def test_network_error_details_are_discarded_without_retry(self):
        code, record = self.run_check(urllib.error.URLError(KEY))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "transport_error")
        self.assertNotIn(KEY, self.output.read_text())
        self.assertEqual(len(self.calls), 1)
    def test_incomplete_http_response_does_not_leak_partial_bytes(self):
        class BrokenResponse(Response):
            def read(self, n):
                raise http.client.IncompleteRead(KEY.encode(), 100)
        result = check.fetch_page(KEY, opener=lambda *args, **kwargs: BrokenResponse(page([])))
        self.assertEqual(result, {"status": "transport_error"})
        self.assertNotIn(KEY, json.dumps(result))
    def test_http_protocol_error_does_not_leak_status_line(self):
        code, record = self.run_check(http.client.BadStatusLine(KEY))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "transport_error")
        self.assertNotIn(KEY, self.output.read_text())
    def test_deep_json_is_rejected_without_traceback(self):
        response = Response(raw=b"[" * 2000 + b"0" + b"]" * 2000)
        result = check.fetch_page(KEY, opener=lambda *args, **kwargs: response)
        self.assertIn(result["status"], {"invalid_json", "invalid_model_page"})
    def test_redirect_handler_never_returns_followup_request(self):
        req = check.urllib.request.Request(check.API_ROOT + "/v1/models")
        self.assertIsNone(check.NoRedirect().redirect_request(req, None, 302, "", {}, "https://elsewhere.invalid"))
    def test_response_size_is_bounded(self):
        response = Response(raw=b"x" * (check.MAX_RESPONSE_BYTES + 2))
        result = check.fetch_page(KEY, opener=lambda *args, **kwargs: response)
        self.assertEqual(response.read_limit, check.MAX_RESPONSE_BYTES + 1)
        self.assertEqual(result["status"], "response_too_large")
    def test_malformed_response_does_not_persist_raw_body(self):
        result = check.fetch_page(KEY, opener=lambda *args, **kwargs: Response(raw=KEY.encode()))
        self.assertEqual(result["status"], "invalid_json")
        self.assertNotIn(KEY, json.dumps(result))
    def test_missing_key_non_terminal_no_echo_and_no_network(self):
        stream = mock.Mock()
        stream.isatty.return_value = False
        reader = mock.Mock(side_effect=AssertionError("must not request echoed input"))
        code, record = check.run(self.output, environ={}, stdin=stream, secret_reader=reader, opener=self.sender())
        self.assertEqual(code, 2)
        self.assertEqual(record["status"], "interactive_terminal_required")
        reader.assert_not_called()
        self.assertEqual(self.calls, [])
    def test_existing_env_does_not_prompt(self):
        reader = mock.Mock(side_effect=AssertionError("must not prompt"))
        key, source = check.acquire_key({"ANTHROPIC_API_KEY": KEY}, secret_reader=reader)
        self.assertEqual((key, source), (KEY, "environment"))
    def test_hidden_terminal_input_is_not_added_to_environment(self):
        env = {}
        stream = mock.Mock()
        stream.isatty.return_value = True
        key, source = check.acquire_key(env, stream, lambda prompt: KEY)
        self.assertEqual((key, source), (KEY, "hidden_terminal_input"))
        self.assertEqual(env, {})
    def test_getpass_fallback_warning_aborts_before_input(self):
        stream = mock.Mock()
        stream.isatty.return_value = True
        def reader(prompt):
            warnings.warn("Cannot control echo.", check.getpass.GetPassWarning)
            raise AssertionError("echo fallback must never execute")
        with self.assertRaisesRegex(check.CheckError, "hidden_input_unavailable_or_cancelled"):
            check.acquire_key({}, stream, reader)
    def test_control_character_credential_never_reaches_network(self):
        code, record = check.run(self.output, environ={"ANTHROPIC_API_KEY": KEY + "\n"}, opener=self.sender())
        self.assertEqual(code, 2)
        self.assertEqual(record["status"], "invalid_credential_format")
        self.assertEqual(self.calls, [])
    def test_secret_in_output_path_rejected_before_file_creation(self):
        output = Path(self.temp.name) / (KEY + ".json")
        with self.assertRaisesRegex(check.CheckError, "invalid_output_path"):
            check.run(output, environ={"ANTHROPIC_API_KEY": KEY}, opener=self.sender())
        self.assertFalse(output.exists())
    def test_cli_rejected_path_is_not_printed(self):
        output = Path(self.temp.name) / (KEY + ".json")
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(check.os.environ, {"ANTHROPIC_API_KEY": KEY}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = check.main(["--output", str(output)])
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertNotIn(KEY, stdout.getvalue() + stderr.getvalue())
    def test_no_network_after_page_limit(self):
        with mock.patch.object(check, "MAX_PAGES", 2):
            code, record = self.run_check(page(["claude-a"], True), page(["claude-b"], True))
        self.assertEqual(code, 3)
        self.assertEqual(record["status"], "pagination_limit_reached")
        self.assertEqual(len(self.calls), 2)
    def test_persistence_failure_prevents_network(self):
        with mock.patch.object(check, "save_owned", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_check(page([]))
        self.assertEqual(self.calls, [])
    def test_argument_error_does_not_echo_unknown_secret(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            check.main(["--api-key", KEY])
        self.assertNotIn(KEY, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
