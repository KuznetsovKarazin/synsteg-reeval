#!/usr/bin/env python3
"""Check direct Anthropic model-list access, without generating text.

Python 3.10+; standard library only. No command-line credential, retries,
redirects, balance inference, or raw provider-response persistence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
import warnings

API_ROOT = "https://api.anthropic.com"
API_VERSION = "2026-10-21"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PAGE_SIZE = 100
MAX_PAGES = 20
ID_PATTERN = re.compile(r"claude-[A-Za-z0-9._:-]{1,150}\Z")


class CheckError(Exception):
    """Carries only a constant, non-sensitive error code."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def now():
    return datetime.now(timezone.utc).isoformat()


def acquire_key(environ=None, stdin=None, secret_reader=None):
    env = os.environ if environ is None else environ
    stream = sys.stdin if stdin is None else stdin
    key = env.get("ANTHROPIC_API_KEY")
    source = "environment"
    if not key:
        if not stream.isatty():
            raise CheckError("interactive_terminal_required")
        source = "hidden_terminal_input"
        reader = getpass.getpass if secret_reader is None else secret_reader
        # getpass may otherwise fall back to echoed input on an unsuitable console.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                key = reader("Anthropic API key (hidden input): ")
            except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
                raise CheckError("hidden_input_unavailable_or_cancelled") from None
    if (not isinstance(key, str) or not 20 <= len(key) <= 512
            or any(not 33 <= ord(ch) <= 126 for ch in key)):
        raise CheckError("invalid_credential_format")
    return key, source


def fetch_page(key, after_id=None, opener=None):
    query = {"limit": PAGE_SIZE}
    if after_id is not None:
        if not isinstance(after_id, str) or not ID_PATTERN.fullmatch(after_id) or key in after_id:
            raise CheckError("invalid_pagination_cursor")
        query["after_id"] = after_id
    req = urllib.request.Request(
        API_ROOT + "/v1/models?" + urllib.parse.urlencode(query), method="GET",
        headers={"x-api-key": key, "anthropic-version": API_VERSION,
                 "accept": "application/json"})
    send = urllib.request.build_opener(NoRedirect()).open if opener is None else opener
    try:
        response = send(req, timeout=30)
        with response:
            status = response.status
            if type(status) is not int or not 200 <= status < 300:
                return {"status": "http_error", "http_status": status if type(status) is int else None}
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Do not retain error bodies, arbitrary headers, exception text or request IDs.
        code = exc.code
        exc.close()
        return {"status": "http_error", "http_status": code}
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        return {"status": "transport_error"}
    if len(raw) > MAX_RESPONSE_BYTES:
        return {"status": "response_too_large", "http_status": status}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return {"status": "invalid_json", "http_status": status}
    if (not isinstance(obj, dict) or not isinstance(obj.get("data"), list)
            or len(obj["data"]) > PAGE_SIZE or type(obj.get("has_more")) is not bool):
        return {"status": "invalid_model_page", "http_status": status}
    # Whitelist model IDs only. Reject credential echoes rather than storing
    # raw bodies, display names, cursor text, error messages or arbitrary metadata.
    ids = []
    for item in obj["data"]:
        model = item.get("id") if isinstance(item, dict) else None
        if (not isinstance(model, str) or not ID_PATTERN.fullmatch(model)
                or key in model or "sk-ant-" in model or "sk-or-" in model):
            return {"status": "invalid_or_sensitive_model_id", "http_status": status}
        if model in ids:
            return {"status": "duplicate_model_id", "http_status": status}
        ids.append(model)
    last_id = obj.get("last_id")
    if obj["has_more"] and (not ids or last_id != ids[-1]):
        return {"status": "invalid_pagination_cursor", "http_status": status}
    return {"status": "ok", "http_status": status, "model_ids": ids,
            "has_more": obj["has_more"], "last_id": ids[-1] if ids else None}


def save_owned(path, record):
    """Replace only a path already exclusively reserved by this process."""
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as file:
            temp_name = file.name
            json.dump(record, file, indent=2, ensure_ascii=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def run(output, environ=None, stdin=None, secret_reader=None, opener=None):
    output = Path(output)
    credential_error = None
    try:
        key, source = acquire_key(environ, stdin, secret_reader)
    except CheckError as exc:
        credential_error = str(exc)
        key, source = None, None
    if key and key in str(output.resolve()):
        raise CheckError("invalid_output_path")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "synsteg-claude-access-check-v1", "status": "started",
              "created_at_utc": now(), "api_version": API_VERSION,
              "endpoint": API_ROOT + "/v1/models", "method": "GET",
              "generation_requests_sent": 0, "automatic_retries": 0,
              "credit_balance_checked": False, "inference_verified": False,
              "model_ids": [], "pages": [],
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    if credential_error:
        record.update(status=credential_error, finished_at_utc=now(), exit_code=2)
        save_owned(output, record)
        return 2, record
    record["credential_source"] = source
    after = None
    code = 3
    for number in range(1, MAX_PAGES + 1):
        page = {"page": number, "status": "request_started", "started_at_utc": now()}
        record["pages"].append(page)
        save_owned(output, record)  # Durable ledger before the actual GET.
        result = fetch_page(key, after, opener)
        page.update(status=result["status"], finished_at_utc=now())
        if "http_status" in result:
            page["http_status"] = result["http_status"]
        if result["status"] != "ok":
            record["status"] = result["status"]
            break
        overlap = set(record["model_ids"]) & set(result["model_ids"])
        if overlap:
            record["status"] = "repeated_pagination_results"
            page["status"] = "repeated_pagination_results"
            break
        record["model_ids"].extend(result["model_ids"])
        page["models_received"] = len(result["model_ids"])
        if not result["has_more"]:
            record["status"] = "model_list_access_verified_no_generation"
            code = 0
            break
        after = result["last_id"]
        save_owned(output, record)
    else:
        record["status"] = "pagination_limit_reached"
    record.update(finished_at_utc=now(), exit_code=code, model_count=len(record["model_ids"]))
    save_owned(output, record)
    return code, record


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Invalid arguments. Use --help. Never supply an API key as an argument.\n")


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    output = args.output or Path("reports/claude-access") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10] + ".json")
    try:
        code, record = run(output)
    except CheckError:
        print("Invalid local configuration. No request was sent.", file=sys.stderr)
        return 2
    except FileExistsError:
        print("Output already exists. Choose a new report path.", file=sys.stderr)
        return 2
    except OSError:
        print("Could not save the access report. No further requests will be made.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Keep the report; no automatic retry was performed.", file=sys.stderr)
        return 130
    print("Status: " + record["status"])
    print("Report: " + str(output))
    if code == 0:
        print("Available model IDs:")
        for model in record["model_ids"]:
            print("  " + model)
    print("No generation was requested. Credit balance and inference access remain unverified.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
