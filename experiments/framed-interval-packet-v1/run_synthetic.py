"""Run only after Git freeze; retains every predefined case and receiver variant."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synsteg_audit.framed_interval_packet_v1 import encode, make_envelope
from synsteg_audit.framed_interval_packet_v1_fixtures import PROMPT_IDS, public_bundle
from synsteg_audit.integer_interval_channel_v1 import Config


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def variant_request(name, request, index):
    changed = json.loads(json.dumps(request))
    wire = bytes.fromhex(request["wire_hex"])
    if name == "wrong_key":
        changed["key_hex"] = bytes(value ^ 255 for value in bytes.fromhex(request["key_hex"])).hex()
    elif name == "wrong_nonce":
        changed["envelope"]["nonce_hex"] = (1001 + index).to_bytes(12, "big").hex()
    elif name == "wrong_aad":
        changed["envelope"]["max_wire_bytes"] += 1
    elif name == "tamper_first_or_insert_if_empty":
        changed["wire_hex"] = (bytes([65 + (wire[0] - 65 + 1) % 16]) + wire[1:] if wire else b"A").hex()
    elif name == "truncate_last_byte":
        changed["wire_hex"] = wire[:-1].hex()
    elif name == "append_A":
        changed["wire_hex"] = (wire + b"A").hex()
    elif name != "original":
        raise ValueError("unknown predefined receiver variant")
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--output", default="results/framed-interval-packet-v1-diagnostic")
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.freeze_commit) is None:
        raise SystemExit("exact Git freeze commit required; external coordinator verifies its readback")
    plan = json.loads((ROOT / "experiments/framed-interval-packet-v1/proposed-protocol.json").read_text())
    manifest = json.loads((ROOT / "experiments/framed-interval-packet-v1/freeze-manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        if sha((ROOT / relative).read_bytes()) != expected:
            raise SystemExit("frozen source/resource mismatch: " + relative)
    if importlib.metadata.version("cryptography") != plan["cryptography_version"]:
        raise SystemExit("cryptography version differs from frozen plan")
    output = ROOT / args.output
    if output.exists():
        raise SystemExit("output already exists; automatic repeat/overwrite forbidden")
    output.mkdir(parents=True)
    key = bytes.fromhex(plan["public_test_key_hex"])
    nonce_values = [case["nonce_hex"] for case in plan["cases"]]
    if (len(nonce_values) != len(set(nonce_values))
            or any(type(nonce) is not str or re.fullmatch(r"[0-9a-f]{24}", nonce) is None for nonce in nonce_values)):
        raise SystemExit("matrix must use unique canonical 96-bit nonces")
    write(output / "started.json", {"freeze_commit": args.freeze_commit, "plan_sha256": sha((ROOT / "experiments/framed-interval-packet-v1/proposed-protocol.json").read_bytes()),
          "planned_cases": len(plan["cases"]), "planned_receiver_runs": len(plan["cases"]) * len(plan["receiver_variants"]),
          "external_git_verification": "required from coordinator; this runner validates pinned local file hashes",
          "python": sys.version, "platform": platform.platform(), "cryptography_version": importlib.metadata.version("cryptography")})
    cases, variants = [], []
    for index, case in enumerate(plan["cases"]):
        case_dir = output / "cases" / case["id"]
        write(case_dir / "started.json", {"status": "reserved_outcome_unknown", "case_id": case["id"]})
        started = time.monotonic()
        try:
            bundle = public_bundle(case["profile"])
            envelope = make_envelope(bundle, bytes.fromhex(case["nonce_hex"]), config=Config(**case["interval_config"]),
                                     prompt_ids=PROMPT_IDS, max_payload_bytes=case["max_payload_bytes"],
                                     max_wire_bytes=case["max_wire_bytes"])
            result = encode(bytes.fromhex(case["payload_hex"]), key, envelope, bundle)
        except Exception as exc:
            write(case_dir / "sender_error.json", {"status": "sender_error", "error_type": type(exc).__name__,
                  "message": str(exc), "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started})
            for variant in plan["receiver_variants"]:
                row = {"case_id": case["id"], "variant": variant, "status": "receiver_not_attempted_due_sender_error",
                       "authenticated_payload_matches": False, "wire_changed": None, "receiver_pid": None,
                       "plaintext_returned": False, "attempted": False, "process_spawned": False,
                       "validated_receiver_report": False}
                write(case_dir / "receivers" / variant / "not_attempted.json", row)
                variants.append(row)
            cases.append({"case_id": case["id"], "profile": case["profile"], "sender_status": "sender_error",
                          "sender_reason": type(exc).__name__, "payload_bytes": len(bytes.fromhex(case["payload_hex"])),
                          "public_slot_bytes": case["max_payload_bytes"], "frame_bits": 8 * (case["max_payload_bytes"] + 18),
                          "length_and_tag_overhead_bits": 144,
                          "padding_bits": 8 * (case["max_payload_bytes"] - len(bytes.fromhex(case["payload_hex"]))),
                          "public_nonce_bits": 96, "shared_key_bits": 256,
                          "public_envelope_utf8_bytes": None, "wire_bytes": None, "packet_count": None,
                          "resolved_bits": None, "consumed_bits": None, "pending_bits": None,
                          "original_receiver_status": "receiver_not_attempted_due_sender_error",
                          "authenticated_payload_matches": False})
            continue
        sender = asdict(result)
        sender["wire_hex"] = result.wire.hex()
        del sender["wire"]
        sender["wall_seconds"] = time.monotonic() - started
        write(case_dir / "sender.json", sender)
        write(case_dir / "envelope.json", envelope)
        original = {"wire_hex": result.wire.hex(), "key_hex": key.hex(), "envelope": envelope}
        case_rows = []
        for variant in plan["receiver_variants"]:
            request = variant_request(variant, original, index)
            directory = case_dir / "receivers" / variant
            write(directory / "request.json", request)
            request_bytes = json.dumps(request).encode()
            write(directory / "started.json", {"status": "reserved_outcome_unknown",
                  "stdin_sha256": sha(request_bytes), "request_fields": sorted(request)})
            started = time.monotonic()
            timed_out = False
            spawn_error = None
            try:
                child = subprocess.run([sys.executable, str(ROOT / "experiments/framed-interval-packet-v1/receiver.py")],
                                       input=request_bytes, capture_output=True, timeout=30,
                                       cwd=directory, env={"PYTHONPATH": str(ROOT / "src"), "PYTHONNOUSERSITE": "1",
                                                           "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"})
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                child = SimpleNamespace(returncode=None, stdout=exc.stdout or b"", stderr=exc.stderr or b"")
            except OSError as exc:
                spawn_error = type(exc).__name__
                child = SimpleNamespace(returncode=None, stdout=b"", stderr=str(exc).encode())
            (directory / "stdout.json").write_bytes(child.stdout)
            (directory / "stderr.txt").write_bytes(child.stderr)
            receipt = {"returncode": child.returncode, "wall_seconds": time.monotonic() - started,
                       "timed_out": timed_out, "spawn_error": spawn_error,
                       "stdin_sha256": sha(request_bytes), "request_fields": sorted(request),
                       "stdout_sha256": sha(child.stdout), "stderr_sha256": sha(child.stderr)}
            parsed = None
            if child.returncode == 0:
                try:
                    candidate = json.loads(child.stdout)
                    if (candidate["schema"] != "framed-interval-packet-v1-fresh-receiver"
                            or candidate["input_fields"] != ["envelope", "key_hex", "wire_hex"]
                            or candidate["cryptography_version"] != plan["cryptography_version"]
                            or candidate["denied_events"] or candidate["forbidden_modules_loaded"]
                            or type(candidate["pid"]) is not int or candidate["pid"] <= 0
                            or candidate["pid"] == os.getpid()):
                        raise ValueError("receiver process/schema/guard mismatch")
                    report = candidate["result"]
                    if type(report) is not dict or set(report) != {"status", "reason", "payload_hex",
                            "required_frame_bits", "resolved_bits", "packet_count", "traces"}:
                        raise ValueError("invalid receiver result fields")
                    if report["status"] not in {"authenticated_delivery", "wire_limit_exceeded", "invalid_wire",
                            "construction_refusal", "insufficient_capacity", "authentication_failed", "invalid_frame"}:
                        raise ValueError("unknown receiver status")
                    if (type(report["required_frame_bits"]) is not int
                            or report["required_frame_bits"] != 8 * (request["envelope"]["max_payload_bytes"] + 18)
                            or type(report["resolved_bits"]) is not int or report["resolved_bits"] < 0
                            or type(report["packet_count"]) is not int or report["packet_count"] < 0
                            or type(report["traces"]) is not list):
                        raise ValueError("invalid receiver result counters")
                    delivered = report["status"] == "authenticated_delivery"
                    if delivered:
                        if (report["reason"] is not None or type(report["payload_hex"]) is not str
                                or re.fullmatch(r"[0-9a-f]*", report["payload_hex"]) is None
                                or len(report["payload_hex"]) % 2
                                or len(report["payload_hex"]) > request["envelope"]["max_payload_bytes"] * 2):
                            raise ValueError("invalid authenticated payload representation")
                    elif report["payload_hex"] is not None or type(report["reason"]) is not str:
                        raise ValueError("failure returned plaintext or lacks a reason")
                    parsed = candidate
                except (ValueError, KeyError, TypeError) as exc:
                    receipt["receiver_output_validation_error"] = type(exc).__name__
            write(directory / "process.json", receipt)
            status = "receiver_process_failed" if parsed is None else parsed["result"]["status"]
            payload_match = (parsed is not None and status == "authenticated_delivery"
                             and parsed["result"]["payload_hex"] == case["payload_hex"])
            row = {"case_id": case["id"], "variant": variant, "status": status,
                   "authenticated_payload_matches": payload_match,
                   "wire_changed": request["wire_hex"] != original["wire_hex"],
                   "receiver_pid": None if parsed is None else parsed["pid"],
                   "plaintext_returned": parsed is not None and parsed["result"]["payload_hex"] is not None,
                   "attempted": True, "process_spawned": spawn_error is None,
                   "validated_receiver_report": parsed is not None}
            case_rows.append(row)
            variants.append(row)
        base = case_rows[0]
        cases.append({"case_id": case["id"], "profile": case["profile"], "sender_status": result.status,
                      "sender_reason": result.reason, "payload_bytes": len(bytes.fromhex(case["payload_hex"])),
                      "public_slot_bytes": case["max_payload_bytes"], "frame_bits": result.required_frame_bits,
                      "length_and_tag_overhead_bits": 144,
                      "padding_bits": 8 * (case["max_payload_bytes"] - len(bytes.fromhex(case["payload_hex"]))),
                      "public_nonce_bits": 96, "shared_key_bits": 256,
                      "public_envelope_utf8_bytes": len(json.dumps(envelope, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()),
                      "wire_bytes": len(result.wire), "packet_count": result.packet_count,
                      "resolved_bits": result.resolved_bits, "consumed_bits": result.consumed_bits,
                      "pending_bits": result.pending_bits, "original_receiver_status": base["status"],
                      "authenticated_payload_matches": base["authenticated_payload_matches"]})
    successful_ids = {row["case_id"] for row in cases if row["authenticated_payload_matches"]}
    negatives = [row for row in variants if row["variant"] != "original"]
    conditional = [row for row in negatives if row["case_id"] in successful_ids]
    summary = {"schema": "framed-interval-packet-v1-synthetic-summary", "freeze_commit": args.freeze_commit,
               "case_denominator": len(cases), "receiver_run_denominator": len(variants),
               "receiver_process_attempts": sum(row["attempted"] for row in variants),
               "receiver_processes_spawned": sum(row["process_spawned"] for row in variants),
               "validated_receiver_reports": sum(row["validated_receiver_report"] for row in variants),
               "receiver_not_attempted": sum(not row["attempted"] for row in variants),
               "authenticated_matching_deliveries": len(successful_ids),
               "nonempty_case_denominator": sum(row["payload_bytes"] > 0 for row in cases),
               "nonempty_authenticated_matching_deliveries": sum(row["payload_bytes"] > 0 and row["authenticated_payload_matches"] for row in cases),
               "negative_receiver_denominator_all_cases": len(negatives),
               "negative_receiver_denominator_original_successes": len(conditional),
               "unexpected_negative_authentication_all_cases": sum(row["plaintext_returned"] for row in negatives),
               "unexpected_negative_authentication_original_successes": sum(row["plaintext_returned"] for row in conditional),
               "verified_negative_rejections_all_cases": sum(row["validated_receiver_report"] and not row["plaintext_returned"] for row in negatives),
               "unassessed_negative_variants": sum(not row["validated_receiver_report"] for row in negatives),
               "cases": cases, "receiver_variants": variants,
               "neural_calls": 0, "tokenizer_calls": 0, "paid_calls": 0,
               "scientific_scope": "finite artificial public codec; no LLM delivery, language quality or steganographic security claim"}
    write(output / "summary.json", summary)
    write(output / "manifest.json", {"files_sha256": {str(path.relative_to(output)): sha(path.read_bytes())
          for path in sorted(output.rglob("*")) if path.is_file()}})
    print(json.dumps({k: summary[k] for k in ("case_denominator", "receiver_run_denominator", "authenticated_matching_deliveries", "unexpected_negative_authentication_all_cases")}))


if __name__ == "__main__":
    main()
