#!/usr/bin/env python3
"""Freeze then audit the authorized exploratory guard policy; no model calls.

Three dependent development cohorts remain separate. No closed-split input,
outcome-based source selection, or contextual-quality efficacy claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import shared_dictionary as base
from synsteg_audit import invariant_context as guarded

SCHEMA = "synsteg-invariant-context-development-v1"
GUARD_PHRASES = ["over time", "this study", "in terms", "dark matter", "drawn from"]
METHODS = ("base_v1", "wrapper_empty", "wrapper_guard")
ARMS = (("plain_length16", 8), ("plain_length16", 16), ("aes_siv", 8))
COUNTS = {"source_excerpts": 100, "historical_covers": 4, "grounded_covers": 20}
SOURCE_PROMPTS = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
BASELINE_RUN = "results/recovery-development-v1/linux_snapshot/models/qwen05"
LENGTH_RUN = "results/recovery-length-probe-v1/run"
GROUNDED_RUN = "results/grounded-development-v2/run"
DICTIONARY = "data/shared_dictionary_recovery_v1.json"
CODE_FILES = ("scripts/audit_invariant_context_development.py", "src/synsteg_audit/invariant_context.py",
              "src/synsteg_audit/shared_dictionary.py", "src/synsteg_audit/__init__.py", "scripts/run_shared_dictionary_pilot.py")
PROVENANCE_KEYS = ("title", "page_id", "revision_id", "revision_url", "history_url", "attribution", "license", "license_url", "excerpt_sha256")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def formatted(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _load_run(root, relative, inventory):
    directory = root / relative
    paths = [directory / "run.json", *sorted((directory / "attempts").glob("*.json"))]
    for path in paths:
        inventory[path.relative_to(root).as_posix()] = sha(path.read_bytes())
    plan = read(directory / "run.json")
    if plan.get("schema") != "synsteg-budget-generation-run-v1" or plan.get("split") != "development":
        raise ValueError("Only development generation plans are accepted")
    selected = plan.get("selected_records")
    if not isinstance(selected, list) or plan.get("selected_records_count") != len(selected):
        raise ValueError("Invalid planned-record denominator")
    by_id = {}
    for row in selected:
        if row.get("split") != "development" or row["record_id"] in by_id:
            raise ValueError("Closed split or duplicate planned record")
        by_id[row["record_id"]] = row
    attempts = {}
    for path in paths[1:]:
        attempt = read(path)
        record = attempt.get("record", {})
        identity = record.get("record_id")
        if identity not in by_id or identity in attempts or record != by_id[identity]:
            raise ValueError("Attempt identity differs from frozen generation plan")
        if attempt.get("schema") != "synsteg-budget-generation-attempt-v1" or any(
                attempt.get(key) != plan.get(key) for key in ("provider", "model")):
            raise ValueError("Attempt provider/model/schema mismatch")
        attempts[identity] = (attempt, path.relative_to(root).as_posix())
    return plan, attempts


def collect_inputs(root):
    """Selection reads saved plans, never review ratings or capacity outcomes."""
    root = Path(root)
    source = root / SOURCE_PROMPTS
    inventory = {SOURCE_PROMPTS: sha(source.read_bytes())}
    sources = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sources) != 100 or any(row.get("split") != "development" for row in sources):
        raise ValueError("Exactly 100 development-only source excerpts are required")
    by_group = {row["source_group_id"]: row for row in sources}
    if len(by_group) != 100 or len({row["record_id"] for row in sources}) != 100:
        raise ValueError("Development source records/groups must be distinct")
    rows = []
    for index, row in enumerate(sources):
        if not isinstance(row.get("excerpt"), str) or sha(row["excerpt"].encode()) != row.get("excerpt_sha256"):
            raise ValueError("Source excerpt hash mismatch")
        rows.append({"cohort": "source_excerpts", "variant": "source_excerpt", "cohort_index": index,
            "audit_record_id": "source:" + row["record_id"], "record_id": row["record_id"],
            "source_group_id": row["source_group_id"], "split": "development", "text": row["excerpt"],
            "text_sha256": sha(row["excerpt"].encode()), "source_file": SOURCE_PROMPTS,
            "source_provenance": {key: row[key] for key in PROVENANCE_KEYS if key in row},
            "input_status": "source_available"})
    baseline, baseline_attempts = _load_run(root, BASELINE_RUN, inventory)
    length, length_attempts = _load_run(root, LENGTH_RUN, inventory)
    grounded, grounded_attempts = _load_run(root, GROUNDED_RUN, inventory)
    if len(baseline["selected_records"]) < 2 or len(length["selected_records"]) != 2 or len(grounded["selected_records"]) != 20:
        raise ValueError("The fixed historical two-pair / grounded-20 selection is unavailable")
    if [(r["record_id"], r["source_group_id"]) for r in baseline["selected_records"][:2]] != [
            (r["record_id"], r["source_group_id"]) for r in length["selected_records"]]:
        raise ValueError("Historical baseline/length variants are not the same first two sources")
    if len({r["source_group_id"] for r in baseline["selected_records"][:2]}) != 2:
        raise ValueError("Historical cohort must retain exactly two distinct source groups")
    if len({r["source_group_id"] for r in grounded["selected_records"]}) != 20:
        raise ValueError("Grounded cohort must retain 20 distinct source groups")
    for cohort, variant, plan_rows, attempts in (
            ("historical_covers", "initial", baseline["selected_records"][:2], baseline_attempts),
            ("historical_covers", "length_probe", length["selected_records"], length_attempts),
            ("grounded_covers", "grounded_v2", grounded["selected_records"], grounded_attempts)):
        for record in plan_rows:
            if record["source_group_id"] not in by_group:
                raise ValueError("Cover source is outside the frozen development panel")
            parent = by_group[record["source_group_id"]]
            if parent["record_id"] != record["record_id"]:
                raise ValueError("Cover record ID and source-group identity disagree")
            entry = attempts.get(record["record_id"])
            attempt, path = entry if entry else ({"status": "not_attempted"}, None)
            result = attempt.get("result") or {}
            text = result.get("cover_text") if attempt["status"] == "completed" else None
            if text is not None and not isinstance(text, str):
                raise ValueError("Cover text must be a string")
            rows.append({"cohort": cohort, "variant": variant,
                "cohort_index": sum(r["cohort"] == cohort for r in rows),
                "audit_record_id": variant + ":" + record["record_id"], "record_id": record["record_id"],
                "source_group_id": record["source_group_id"], "split": "development", "text": text,
                "text_sha256": sha(text.encode()) if text is not None else None, "source_file": path,
                "source_provenance": {key: parent[key] for key in PROVENANCE_KEYS if key in parent},
                "input_status": attempt["status"], "generation_stop_reason": result.get("stop_reason"),
                "generation_hit_output_limit": result.get("hit_output_limit")})
    if dict(Counter(row["cohort"] for row in rows)) != COUNTS:
        raise ValueError("Cohort counts changed")
    if any(sha((root / name).read_bytes()) != expected for name, expected in inventory.items()):
        raise ValueError("Input files changed during collection")
    return rows, inventory


def prepare(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Use a new result namespace; preparation never overwrites an existing freeze")
    rows, inventory = collect_inputs(root)
    dictionary_bytes = (root / DICTIONARY).read_bytes()
    dictionary = base.PublicDictionary.from_dict(json.loads(dictionary_bytes))
    policy = guarded.ProtectedPhrasePolicy.from_phrases(dictionary, GUARD_PHRASES,
        version="observed-v1-errors-development-guard-v1")
    empty = guarded.ProtectedPhrasePolicy.from_phrases(dictionary, [], version="empty-policy-development-control-v1")
    inputs = b"".join(canonical(row) + b"\n" for row in rows)
    frozen = {"inputs.jsonl": inputs, "public_dictionary.json": dictionary_bytes,
              "guard_policy.json": formatted(policy.to_dict()), "empty_policy.json": formatted(empty.to_dict())}
    protocol = {"schema": SCHEMA, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post_observation_exploratory_development", "policy_origin": "Only the five phrases authorized after observed v1 failures",
        "approved_phrases": GUARD_PHRASES, "dictionary_intervention": "recovery_v1_only; strict recovery_v2 is not used",
        "public_dictionary_sha256": dictionary.sha256, "dictionary_file_sha256": sha(dictionary_bytes),
        "guard_policy_sha256": policy.sha256, "empty_policy_sha256": empty.sha256,
        "source_files": inventory, "code_files": {name: sha((root / name).read_bytes()) for name in CODE_FILES},
        "frozen_files": {name: sha(data) for name, data in frozen.items()},
        "cohorts": {cohort: {"planned_records": count,
            "distinct_source_groups": len({row["source_group_id"] for row in rows if row["cohort"] == cohort})}
            for cohort, count in COUNTS.items()},
        "cohort_dependence": "All cohorts reuse the 100 development source groups; historical four covers reuse two groups. Do not pool independent N.",
        "selection": "All 100 source excerpts; first two planned initial covers and their two length variants; all 20 grounded planned covers. No review/capacity filtering.",
        "methods": list(METHODS), "arms": [{"framing": f, "payload_bits": b} for f, b in ARMS],
        "base_protocols": {f: base.Protocol(min_group_size=1, max_bits_per_slot=1, framing=f).to_dict() for f in ("plain_length16", "aes_siv")},
        "wrapper_protocols": {f: guarded.ContextProtocol(framing=f).to_dict() for f in ("plain_length16", "aes_siv")},
        "fixture_key_rule": "SHA256 of UTF-8 PUBLIC-INVARIANT-CONTEXT-KEY-v1 + NUL + source_group_id; not secret",
        "fixture_payload_rule": "SHA256 of UTF-8 PUBLIC-INVARIANT-CONTEXT-PAYLOAD-v1 + NUL + source_group_id; take first payload_bytes",
        "receiver": "Fresh process for every deliverable frame; only key/stego/public resources, guarded file/socket/subprocess I/O",
        "failures": "Keep unavailable inputs, domain rejection, capacity and decoder failures in each cohort's planned denominator; no retries",
        "textual_comparison": "Retain texts for inspection; policy/key-domain/normalization changes can change all other synonym choices, including empty wrapper versus base",
        "quality_effect_attribution": False, "population_inference": False, "closed_test_used": False,
        "runtime_at_preparation": {"python": sys.version, "cryptography": importlib.metadata.version("cryptography")},
        "new_model_calls": 0, "paid_calls": 0, "actual_codec_encoding_calls_during_preparation": 0}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in frozen.items():
        exclusive(output / name, data)
    # Written last: this immutable artifact must exist before any actual encoding.
    exclusive(output / "protocol.json", formatted(protocol))
    return {"status": "frozen_before_encoding", "protocol_sha256": sha((output / "protocol.json").read_bytes()),
            "guard_policy_sha256": policy.sha256, "planned_records_by_cohort": COUNTS}


def verify_freeze(root, output):
    protocol = read(output / "protocol.json")
    if protocol.get("schema") != SCHEMA or protocol.get("approved_phrases") != GUARD_PHRASES:
        raise ValueError("Unknown or changed exploratory protocol")
    for location, names in ((root, protocol["code_files"]), (root, protocol["source_files"]), (output, protocol["frozen_files"])):
        for name, expected in names.items():
            path = (location / name).resolve()
            if not path.is_relative_to(location.resolve()) or sha(path.read_bytes()) != expected:
                raise ValueError("Frozen file changed: " + name)
    return protocol


def fixture(label, group, size=32):
    return hashlib.sha256(("PUBLIC-INVARIANT-CONTEXT-" + label + "-v1\0" + group).encode()).digest()[:size]


def isolated_receive(root, method, text, key, dictionary, policy, framing):
    if method == "base_v1":
        executable = root / "scripts/run_shared_dictionary_pilot.py"
        protocol = base.Protocol(min_group_size=1, max_bits_per_slot=1, framing=framing)
        request = {"key_hex": key.hex(), "stego_text": text, "public_dictionary": dictionary.to_dict(),
                   "public_protocol": protocol.to_dict()}
    else:
        executable = root / "src/synsteg_audit/invariant_context.py"
        protocol = guarded.ContextProtocol(framing=framing)
        request = {"key_hex": key.hex(), "stego_text": text, "public_dictionary": dictionary.to_dict(),
                   "public_policy": policy.to_dict(), "public_protocol": protocol.to_dict()}
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP") if name in os.environ}
    environment.update(PATH=os.defpath, PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="invariant-context-receiver-") as directory:
        child = subprocess.run([sys.executable, "-I", str(executable.resolve()), "receiver"],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env=environment, timeout=60, check=False)
    response = json.loads(child.stdout)
    if (child.returncode != 0 or response.get("status") != "decoded" or response.get("input_keys") != sorted(request)
            or response.get("io_guard_self_checks") != {"file": True, "socket": True, "subprocess": True}):
        raise ValueError("Independent receiver rejected or lacked isolation evidence")
    expected = ({"public_dictionary_sha256": dictionary.sha256, "public_protocol_sha256": protocol.sha256}
                if method == "base_v1" else {"dictionary_sha256": dictionary.sha256,
                    "policy_sha256": policy.sha256, "protocol_sha256": protocol.sha256})
    if any(response.get(name) != value for name, value in expected.items()):
        raise ValueError("Independent receiver public-resource identities differ")
    return bytes.fromhex(response["payload_hex"]), {"actual_returncode": child.returncode,
        "input_keys": response["input_keys"], "io_guard_self_checks": response["io_guard_self_checks"],
        "response_sha256": sha(child.stdout), "stderr_sha256": sha(child.stderr)}


def evaluate_record(root, row, dictionary, policies):
    text = row["text"]
    key = fixture("KEY", row["source_group_id"])
    result = {**row, "words": len(text.split()) if text is not None else None,
              "public_fixture_key_hex": key.hex(), "methods": {}}
    for method in METHODS:
        entry = {"capacity_bits": None, "protected_slots": 0, "protected_tokens": [], "arms": {}}
        policy = policies.get(method)
        availability = "available"
        if not text or row.get("generation_stop_reason") == "refusal":
            availability = "unavailable_input"
        else:
            try:
                if method == "base_v1":
                    analysis = base.analyze_text(text, dictionary, base.Protocol(min_group_size=1, max_bits_per_slot=1))
                else:
                    analysis = guarded.analyze_text(text, dictionary, policy)
                    tokens = text.split()
                    entry["protected_slots"] = analysis.protected_slot_count
                    entry["protected_tokens"] = [{"token_index": i, "token": tokens[i]} for i in analysis.protected_token_indices]
                entry["capacity_bits"] = analysis.capacity_bits
            except base.CodecError as error:
                availability = "invalid_input"
                entry["error_type"] = type(error).__name__
        entry["input_status"] = availability
        for framing, bits in ARMS:
            arm = {"payload_bits": bits, "framing": framing, "full_success": False, "encoding_called": False}
            payload = fixture("PAYLOAD", row["source_group_id"], bits // 8)
            arm["payload_hex"] = payload.hex()
            needed = bits + (16 if framing == "plain_length16" else 144)
            if availability != "available":
                arm["status"] = availability
            elif entry["capacity_bits"] < needed:
                arm.update(status="insufficient_capacity", required_frame_bits=needed)
            else:
                arm["encoding_called"] = True
                try:
                    if method == "base_v1":
                        encoded = base.encode(text, payload, key, dictionary,
                            base.Protocol(min_group_size=1, max_bits_per_slot=1, framing=framing))
                    else:
                        encoded = guarded.encode(text, payload, key, dictionary, policy, guarded.ContextProtocol(framing=framing))
                    tokens = encoded.stego_text.split()
                    preserved = all(tokens[p["token_index"]] == p["token"] for p in entry["protected_tokens"])
                    decoded, receiver = isolated_receive(root, method, encoded.stego_text, key, dictionary, policy, framing)
                    arm.update(status="decoded" if decoded == payload and preserved else "payload_or_protection_mismatch",
                        full_success=decoded == payload and preserved, stego_text=encoded.stego_text,
                        stego_text_sha256=sha(encoded.stego_text.encode()), decoded_payload_hex=decoded.hex(),
                        protected_spellings_preserved=preserved, changed_slots=encoded.changed_slots, receiver=receiver)
                except (base.CodecError, ValueError, OSError, subprocess.SubprocessError) as error:
                    arm.update(status="codec_or_receiver_failure", error_type=type(error).__name__)
            entry["arms"][f"{framing}__{bits}"] = arm
        result["methods"][method] = entry
    capacities = {m: result["methods"][m]["capacity_bits"] for m in METHODS}
    if capacities["wrapper_empty"] is not None and capacities["wrapper_empty"] != capacities["base_v1"]:
        raise ValueError("Empty wrapper changed capacity")
    if capacities["wrapper_guard"] is not None and capacities["wrapper_guard"] > capacities["base_v1"]:
        raise ValueError("Guard increased capacity")
    return result


def summarize(rows):
    summaries, examples = {}, []
    for cohort, expected in COUNTS.items():
        subset = [row for row in rows if row["cohort"] == cohort]
        if len(subset) != expected:
            raise ValueError("Incomplete cohort cannot be summarized as completed")
        summary = {"planned_records": expected, "distinct_source_groups": len({r["source_group_id"] for r in subset}), "methods": {}}
        for method in METHODS:
            entries = [row["methods"][method] for row in subset]
            capacities = [e["capacity_bits"] for e in entries if e["capacity_bits"] is not None]
            arms = {}
            for framing, bits in ARMS:
                name = f"{framing}__{bits}"
                counts = dict(Counter(entry["arms"][name]["status"] for entry in entries))
                successes = sum(entry["arms"][name]["full_success"] for entry in entries)
                arms[name] = {"planned": expected, "full_success": successes, "statuses": counts,
                              "fraction_all_planned": successes / expected}
            summary["methods"][method] = {"available_capacity_records": len(capacities),
                "capacity_bits": {"min": min(capacities), "median": statistics.median(capacities), "max": max(capacities),
                                  "mean": statistics.mean(capacities)} if capacities else None,
                "records_with_protected_slots": sum(e["protected_slots"] > 0 for e in entries),
                "protected_slots_total": sum(e["protected_slots"] for e in entries), "arms": arms}
        summaries[cohort] = summary
        for row in subset:
            guard = row["methods"]["wrapper_guard"]
            delivered = [name for name, arm in guard["arms"].items() if arm["full_success"]]
            if guard["protected_slots"] and delivered:
                examples.append({"cohort": cohort, "audit_record_id": row["audit_record_id"], "words": row["words"],
                    "protected_tokens": guard["protected_tokens"], "delivered_arms": delivered,
                    "description": "Mechanical preservation example retained from complete cohort, not evidence of general quality improvement"})
    return {"schema": SCHEMA + "-summary", "cohorts": summaries,
        "cohorts_must_not_be_pooled_as_independent_n": True, "mechanical_protection_examples": examples,
        "population_inference": False, "quality_metrics": "not_computed", "quality_effect_attribution": False,
        "test_split_opened": False, "new_model_calls": 0, "paid_calls": 0,
        "interpretation": ["Protection preserves selected source token spellings; it may reduce usable frame capacity.",
            "Public policy was chosen after earlier development failures; these are exploratory observations.",
            "Empty wrapper controls the presence of a separate protocol, but policy hash and virtual normalization also change synonym choices elsewhere.",
            "Roundtrip success does not establish contextual correctness, factual fidelity, naturalness or steganographic secrecy."]}


def execute(root, output):
    root, output = Path(root), Path(output)
    protocol = verify_freeze(root, output)
    protocol_sha = sha((output / "protocol.json").read_bytes())
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    dictionary = base.PublicDictionary.from_dict(read(output / "public_dictionary.json"))
    policies = {"wrapper_guard": guarded.ProtectedPhrasePolicy.from_dict(read(output / "guard_policy.json"), dictionary),
                "wrapper_empty": guarded.ProtectedPhrasePolicy.from_dict(read(output / "empty_policy.json"), dictionary)}
    inputs = [json.loads(line) for line in (output / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    if (dict(Counter(row.get("cohort") for row in inputs)) != COUNTS
            or len({row.get("audit_record_id") for row in inputs}) != sum(COUNTS.values())
            or any(row.get("split") != "development" for row in inputs)):
        raise ValueError("Execution requires the complete separate development cohorts")
    rows = []
    with (evaluation / "records.jsonl").open("xb") as stream:
        for item in inputs:
            row = evaluate_record(root, item, dictionary, policies)
            rows.append(row)
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    verify_freeze(root, output)
    if sha((output / "protocol.json").read_bytes()) != protocol_sha:
        raise ValueError("Protocol changed during execution")
    report = summarize(rows)
    report["protocol_sha256"] = protocol_sha
    report["records_sha256"] = sha((evaluation / "records.jsonl").read_bytes())
    exclusive(evaluation / "summary.json", formatted(report))
    exclusive(evaluation / "execution.json", formatted({"schema": SCHEMA + "-execution", "status": "completed",
        "protocol_sha256_before_encoding": protocol_sha, "protocol_prepared_at_utc": protocol["prepared_at_utc"],
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - start, "processed_records": len(rows),
        "actual_encoding_calls": sum(a["encoding_called"] for row in rows for m in row["methods"].values() for a in m["arms"].values()),
        "python": sys.version, "cryptography": importlib.metadata.version("cryptography"),
        "automatic_retries": 0, "new_model_calls": 0, "paid_calls": 0}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/invariant-context-development-v1")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        result = prepare(ROOT, args.output)
    elif args.stage == "run":
        result = execute(ROOT, args.output)
        result = {"status": "completed", "cohorts": {c: v["planned_records"] for c, v in result["cohorts"].items()},
                  "mechanical_protection_examples": len(result["mechanical_protection_examples"]), "paid_calls": 0}
    else:
        verify_freeze(ROOT, args.output)
        result = {"status": "freeze_verified", "protocol_sha256": sha((args.output / "protocol.json").read_bytes())}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
