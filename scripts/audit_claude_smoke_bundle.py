#!/usr/bin/env python3
"""Read-only audit of a Claude smoke ZIP; no API calls or file writes.

Usage: python scripts/audit_claude_smoke_bundle.py PATH_TO_ZIP
Source hashes are compared with the local runner/helper beside this script.
Later edits to those sources can therefore cause a source-hash check to fail.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def audit(path):
    checks = []
    output = {
        "schema": "synsteg-claude-smoke-bundle-audit-v1",
        "auditor_sha256": sha256(Path(__file__).read_bytes()),
        "checks": checks,
        "scope": "Internal consistency of user-supplied records; no independent API call, invoice verification, quality ranking, or balance/quota check.",
        "source_hash_scope": "Compare plan hashes with current local scripts, normalized to LF and CRLF. Later source edits may invalidate this comparison; hashes are not provider signatures.",
        "response_hash_scope": "Sorted compact JSON after credential redaction, not original wire bytes.",
    }

    def check(condition, name):
        checks.append({"check": name, "pass": bool(condition)})

    try:
        raw_zip = path.read_bytes()
        output.update(zip_bytes=len(raw_zip), zip_sha256=sha256(raw_zip))
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            expected = {"plan.json", "summary.json", "attempt-01.json", "attempt-02.json", "attempt-03.json"}
            check(len(names) == len(set(names)) == 5, "five unique members")
            check(set(names) == expected, "exact member names")
            if set(names) != expected or len(names) != 5:
                raise ValueError("unexpected archive structure")
            if any(info.file_size > 2 * 1024 * 1024 for info in archive.infolist()):
                raise ValueError("oversized member")
            check(archive.testzip() is None, "CRC")
            raw = {name: archive.read(name) for name in names}
        sensitive = any(re.search(rb"sk-(?:ant|or)-[A-Za-z0-9_-]{12,}", value)
                        for value in raw.values())
        check(not sensitive, "no recognizable credentials")
        if sensitive:
            raise ValueError("possible credential; do not print record values")
        data = {name: json.loads(value) for name, value in raw.items()}
        output["file_sha256"] = {name: sha256(value) for name, value in raw.items()}
        plan, summary = data["plan.json"], data["summary.json"]
        hashes = {}
        for filename, key in (("run_claude_smoke.py", "runner_sha256"),
                              ("check_claude_access.py", "access_helper_sha256")):
            source = Path(__file__).with_name(filename).read_bytes()
            lf = source.replace(b"\r\n", b"\n")
            hashes[filename] = {
                "local_raw": sha256(source), "lf": sha256(lf),
                "crlf": sha256(lf.replace(b"\n", b"\r\n")),
                "archive_claim": plan[key],
            }
            match = [kind for kind in ("lf", "crlf") if hashes[filename][kind] == plan[key]]
            hashes[filename]["matching_line_endings"] = match
            check(bool(match), filename + " source hash matches LF or CRLF")
        output["source_hashes"] = hashes
        check(summary["status"] == "three_generations_recorded" and summary["exit_code"] == 0,
              "successful summary")
        check(summary["generation_requests_attempted"] == 3 and summary["automatic_retries"] == 0,
              "three attempts zero retries")
        check(plan["maximum_generation_requests"] == 3 and
              plan["maximum_output_tokens_per_request"] == 512 and
              plan["maximum_output_tokens_total"] == 1536, "plan bounds")
        rows, total, last = [], Decimal(0), None
        if len(plan["models"]) != 3 or len(plan["requests"]) != 3 or len(summary["results"]) != 3:
            raise ValueError("unexpected number of model records")
        for index, model in enumerate(plan["models"], 1):
            attempt = data[f"attempt-{index:02d}.json"]
            response = attempt["response"]
            retained, result = response["retained_response"], attempt["result"]
            usage = retained["usage"]
            canonical = json.dumps(retained, ensure_ascii=True, sort_keys=True,
                                   separators=(",", ":")).encode()
            check(sha256(canonical) == response["retained_response_sha256"], f"{index} retained response hash")
            check(attempt["request_body"] == plan["requests"][index - 1], f"{index} request identical to plan")
            check(attempt["requested_model"] == attempt["request_body"]["model"] == retained["model"] ==
                  result["requested_model"] == result["returned_model"] == model, f"{index} models match")
            check(retained["id"] == result["response_id"] and response["request_id"] == result["request_id"],
                  f"{index} ids match")
            check(usage == result["usage"], f"{index} usage match")
            check(attempt["status"] == "generation_recorded" and attempt["generation_outcome_known"] and
                  response["status"] == "http_success" and response["http_status"] == 200, f"{index} success state")
            check(retained["stop_reason"] == result["stop_reason"] == "end_turn" and
                  not result["hit_output_limit"] and bool(result["text"].strip()), f"{index} complete text")
            check(not response["credential_redacted"], f"{index} no credential redaction")
            check(attempt["automatic_retries"] == 0 and attempt["method"] == "POST" and
                  attempt["endpoint"] == "https://api.anthropic.com/v1/messages", f"{index} request type")
            check(attempt["api_version"] == plan["api_version"] == "2026-10-21", f"{index} api version")
            check(result["text"] == "".join(block["text"] for block in retained["content"]
                                           if block["type"] == "text").strip(), f"{index} output text match")
            check(0 <= usage["output_tokens"] <= 512 and 0 <= usage["input_tokens"] <= 4096,
                  f"{index} token allowance")
            check(usage["cache_read_input_tokens"] == usage["cache_creation_input_tokens"] == 0,
                  f"{index} no cache usage")
            check(usage["service_tier"] == usage["speed"] == "standard" and usage["inference_geo"] == "global",
                  f"{index} standard price applicable from record")
            prices = plan["prices_per_million_tokens_usd"][model]
            cost = (Decimal(usage["input_tokens"]) * Decimal(str(prices["input"])) +
                    Decimal(usage["output_tokens"]) * Decimal(str(prices["output"]))) / Decimal(1000000)
            check(cost == Decimal(str(result["estimated_standard_token_cost_usd"])), f"{index} recomputed cost")
            total += cost
            start = datetime.fromisoformat(attempt["started_at_utc"])
            end = datetime.fromisoformat(attempt["finished_at_utc"])
            duration = (end - start).total_seconds()
            check(end >= start and (last is None or start >= last), f"{index} sequential UTC times")
            last = end
            rows.append({"model": model, "input": usage["input_tokens"], "output": usage["output_tokens"],
                         "word_count_whitespace": len(result["text"].split()), "wall_seconds": duration,
                         "elapsed_seconds_record": attempt["elapsed_seconds"],
                         "wall_minus_elapsed_seconds": duration - attempt["elapsed_seconds"],
                         "cost_usd": str(cost), "request_id": response["request_id"]})
            check(summary["results"][index - 1] == {"model": model, "status": "generation_recorded",
                  "attempt_file": f"attempt-{index:02d}.json"}, f"{index} summary result index")
        check(abs(total - Decimal(str(summary["estimated_standard_token_cost_usd"]))) < Decimal("1e-18"),
              "summary cost matches within float rounding")
        check(not plan["test_data"] and not summary["test_data"], "development only")
        output.update(rows=rows, input_total=sum(row["input"] for row in rows),
                      output_total=sum(row["output"] for row in rows), cost_total_usd_exact=str(total),
                      batch_wall_seconds=(datetime.fromisoformat(summary["finished_at_utc"]) -
                                          datetime.fromisoformat(summary["created_at_utc"])).total_seconds())
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError,
            InvalidOperation, zipfile.BadZipFile) as error:
        # Do not echo malformed response text, unexpected values, or paths.
        output["audit_error"] = type(error).__name__
    output["passed_checks"] = sum(item["pass"] for item in checks)
    output["total_checks"] = len(checks)
    output["issues"] = [item["check"] for item in checks if not item["pass"]]
    output["status"] = "pass" if not output["issues"] and "audit_error" not in output else "fail"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    result = audit(args.zip_path)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
