#!/usr/bin/env python3
"""One bounded development probe per three Claude models (Python 3.10+).

No execution without --execute; no retries, redirects or overwrite/resume.
The monetary guard is a planning estimate, not an account billing limit.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile

from check_claude_access import (API_ROOT, API_VERSION, CheckError, NoRedirect,
                                 SafeParser, acquire_key, now, save_owned)

MODELS = ("claude-sonnet-4-6", "claude-sonnet-5-5", "claude-haiku-5-5")
MAX_TOKENS = 512
INPUT_PLANNING_ALLOWANCE = 4096
LIMIT_USD = 0.10
PRICES_PER_MILLION = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
}
SYSTEM = "Write clear English prose. Return only the requested text."
PROMPT = ("Write about 120 words explaining how a community library can organize "
          "a small table for exchanging books. Describe a practical arrangement, "
          "a simple rule for visitors, and a reason the arrangement is useful. "
          "Use ordinary paragraphs, with no heading, list, quotation, or dialogue.")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_RUN = Path("reports/claude-smoke-v1")
SECRET_PATTERN = re.compile(r"sk-(?:ant|or)-[A-Za-z0-9_-]+")


def scrub(value, key):
    if isinstance(value, str):
        # A JSON string can itself contain an escaped JSON representation.
        # Inspect a decoded view, but never persist that view or the secret.
        escaped_view = re.sub(r"\\+u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), value)
        if escaped_view != value and (key in escaped_view or SECRET_PATTERN.search(escaped_view)):
            return "[REDACTED_CREDENTIAL]"
        return SECRET_PATTERN.sub("[REDACTED_CREDENTIAL]", value.replace(key, "[REDACTED_CREDENTIAL]"))
    if isinstance(value, list):
        return [scrub(x, key) for x in value]
    if isinstance(value, dict):
        return {scrub(k, key): scrub(v, key) for k, v in value.items()}
    return value


def request_body(model):
    if model not in MODELS:
        raise CheckError("unregistered_model")
    body = {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "output_config": {"effort": "high"}, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": PROMPT}]}
    if model == "claude-sonnet-5-5":
        # 5.5 rejects thinking:disabled. With no tools, between_tools avoids
        # spending the small development output budget on initial reasoning.
        body["thinking"] = {"type": "between_tools"}
    return body


def plan():
    cost = sum((INPUT_PLANNING_ALLOWANCE * p["input"] + MAX_TOKENS * p["output"]) / 1e6
               for p in PRICES_PER_MILLION.values())
    if cost > LIMIT_USD:
        raise CheckError("planning_budget_exceeded")
    return {"schema": "synsteg-claude-smoke-plan-v1", "purpose": "development_inference_check_not_test_data",
            "api_version": API_VERSION, "models": list(MODELS),
            "requests": [request_body(m) for m in MODELS], "maximum_generation_requests": 3,
            "maximum_output_tokens_per_request": MAX_TOKENS,
            "maximum_output_tokens_total": 3 * MAX_TOKENS,
            "input_token_planning_allowance_per_request": INPUT_PLANNING_ALLOWANCE,
            "estimated_max_cost_usd": round(cost, 10), "planning_limit_usd": LIMIT_USD,
            "price_source": "https://platform.claude.com/docs/en/about-claude/pricing",
            "price_checked_date": "2026-10-24", "prices_per_million_tokens_usd": PRICES_PER_MILLION,
            "cost_scope": "Standard token prices and a conservative input planning allowance; not measured token count or an account-wide billing guarantee. No tools, caching, batch, priority or fast-mode options requested.",
            "automatic_retries": 0, "test_data": False,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "access_helper_sha256": hashlib.sha256(Path(__file__).with_name("check_claude_access.py").read_bytes()).hexdigest()}


def send_once(body, key, opener=None):
    req = urllib.request.Request(API_ROOT + "/v1/messages", method="POST",
        data=json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
        headers={"x-api-key": key, "anthropic-version": API_VERSION,
                 "content-type": "application/json", "accept": "application/json"})
    send = urllib.request.build_opener(NoRedirect()).open if opener is None else opener
    try:
        try:
            response = send(req, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = response.status if hasattr(response, "status") else response.code
            request_id = response.headers.get("request-id")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException):
        return {"status": "transport_error_outcome_unknown"}
    if len(raw) > MAX_RESPONSE_BYTES:
        return {"status": "response_too_large_outcome_unknown"}
    if type(status) is not int:
        return {"status": "invalid_http_status"}
    try:
        original = json.loads(raw.decode("utf-8"))
        retained = scrub(original, key)
        canonical = json.dumps(retained, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {"status": "invalid_json_outcome_unknown", "http_status": status}
    safe_request_id = scrub(request_id, key) if isinstance(request_id, str) else None
    return {"status": "http_success" if 200 <= status < 300 else "http_error",
            "http_status": status,
            "request_id": safe_request_id,
            "retained_response": retained,
            "retained_response_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "hash_scope": "Sorted compact JSON after credential redaction; not original wire bytes",
            "credential_redacted": retained != original or safe_request_id != request_id}


def validate_response(reply, model):
    if reply.get("status") != "http_success":
        return None, reply.get("status", "invalid_transport_result")
    if reply.get("credential_redacted"):
        return None, "credential_echo_redacted_stop"
    obj = reply.get("retained_response")
    if (not isinstance(obj, dict) or obj.get("type") != "message"
            or obj.get("role") != "assistant" or obj.get("model") != model
            or not isinstance(obj.get("id"), str) or not obj["id"]):
        return None, "invalid_message_identity"
    usage = obj.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("input_tokens", "output_tokens")):
        return None, "invalid_token_usage"
    for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        if k in usage and (type(usage[k]) is not int or usage[k] != 0):
            return None, "unexpected_cache_usage_inspect_before_continuing"
    if usage["input_tokens"] > INPUT_PLANNING_ALLOWANCE or usage["output_tokens"] > MAX_TOKENS:
        return None, "token_planning_allowance_exceeded"
    blocks = obj.get("content")
    if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
        return None, "invalid_content"
    text_blocks = [b.get("text") for b in blocks if b.get("type") == "text"]
    if not text_blocks or any(not isinstance(t, str) for t in text_blocks):
        return None, "no_valid_visible_text"
    text = "".join(text_blocks).strip()
    if not text:
        return None, "no_valid_visible_text"
    reason = obj.get("stop_reason")
    if reason not in ("end_turn", "max_tokens"):
        return None, "refusal_or_unexpected_stop"
    prices = PRICES_PER_MILLION[model]
    cost = (usage["input_tokens"] * prices["input"] + usage["output_tokens"] * prices["output"]) / 1e6
    return {"requested_model": model, "returned_model": obj["model"], "response_id": obj["id"],
            "request_id": reply.get("request_id"), "usage": usage, "stop_reason": reason,
            "hit_output_limit": reason == "max_tokens", "text": text,
            "estimated_standard_token_cost_usd": cost}, None


def write_new(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bundle(run_dir, member_names):
    archive = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in member_names:
            z.write(run_dir / name, name)
    return archive


def run(execute=False, run_dir=DEFAULT_RUN, environ=None, stdin=None,
        secret_reader=None, opener=None):
    protocol = plan()
    if not execute:
        return 0, {"status": "preview_only_no_requests", "plan": protocol}
    run_dir = Path(run_dir)
    archive = run_dir.with_suffix(".zip")
    # Check before asking for a key or making any request, including after an
    # interrupted run. A fresh run ID requires a deliberate protocol revision.
    if run_dir.exists() or archive.exists():
        raise CheckError("existing_run_refused_no_requests")
    key, source = acquire_key(environ, stdin, secret_reader)
    if key in str(run_dir.resolve()) or key in str(archive.resolve()):
        raise CheckError("invalid_output_path")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "synsteg-claude-smoke-summary-v1", "status": "started",
               "created_at_utc": now(), "credential_source": source,
               "generation_requests_attempted": 0, "automatic_retries": 0,
               "results": [], "test_data": False,
               "estimated_standard_token_cost_usd": 0.0,
               "cost_estimate_scope": "Validated responses only; failed or unknown attempts may also incur charges. Not an invoice or account billing limit.",
               "cost_estimate_covers_all_attempts": False,
               "billing_balance_verified": False}
    members = ["plan.json", "summary.json"]
    write_new(run_dir / "plan.json", protocol)
    write_new(run_dir / "summary.json", summary)
    code = 3
    try:
        for index, model in enumerate(MODELS, 1):
            name = "attempt-%02d.json" % index
            path = run_dir / name
            attempt = {"schema": "synsteg-claude-smoke-attempt-v1", "status": "request_started",
                       "started_at_utc": now(), "requested_model": model,
                       "api_version": API_VERSION, "method": "POST",
                       "endpoint": API_ROOT + "/v1/messages", "request_body": request_body(model),
                       "automatic_retries": 0, "generation_outcome_known": False}
            write_new(path, attempt)
            members.append(name)
            summary["generation_requests_attempted"] += 1
            save_owned(run_dir / "summary.json", summary)
            started = time.monotonic()
            reply = send_once(attempt["request_body"], key, opener)
            attempt.update(response=reply, finished_at_utc=now(),
                           elapsed_seconds=round(time.monotonic() - started, 6))
            result, failure = validate_response(reply, model)
            attempt["status"] = failure or "generation_recorded"
            attempt["generation_outcome_known"] = result is not None
            if result is not None:
                attempt["result"] = result
                summary["estimated_standard_token_cost_usd"] += result["estimated_standard_token_cost_usd"]
            save_owned(path, attempt)
            summary["results"].append({"model": model, "status": attempt["status"], "attempt_file": name})
            if failure:
                summary["status"] = "stopped_on_first_failure"
                break
            if summary["estimated_standard_token_cost_usd"] > LIMIT_USD:
                summary["status"] = "stopped_on_cost_guard"
                break
        else:
            summary["status"] = "three_generations_recorded"
            summary["cost_estimate_covers_all_attempts"] = True
            code = 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted_no_retry"
        code = 130
    except OSError:
        # Stop before any further request; persistence failure may make the
        # last request unknowable. A later invocation refuses this run.
        summary["status"] = "persistence_failed_no_retry"
        code = 2
    summary.update(finished_at_utc=now(), exit_code=code)
    save_owned(run_dir / "summary.json", summary)
    bundle(run_dir, members)
    return code, summary


def main(argv=None):
    parser = SafeParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send at most three generation requests")
    args = parser.parse_args(argv)
    try:
        code, summary = run(execute=args.execute)
    except CheckError as error:
        print("Not started: " + str(error), file=sys.stderr)
        return 2
    except OSError:
        print("Local persistence failed. Keep existing reports; do not retry by deleting them.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Keep existing reports; no automatic retry.", file=sys.stderr)
        return 130
    print("Status: " + summary["status"])
    if args.execute:
        print("Report directory: reports/claude-smoke-v1")
        print("Shareable archive: reports/claude-smoke-v1.zip")
        print("Generation requests attempted: " + str(summary["generation_requests_attempted"]))
    else:
        print("Preview: three models, at most 1536 output tokens; no requests sent.")
        print("Estimated planning cost: $%.8f (not an account billing limit)." % summary["plan"]["estimated_max_cost_usd"])
        print("Use --execute to run this development probe once.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
