#!/usr/bin/env python3
"""Audit recovery-v1 development covers with independent shared-codec receivers.

This is a development diagnostic, not a detector, semantic-quality assessment,
or reproduction of the lost pilot. Planned samples come from run.json, never
from whichever generations happened to succeed. No network/model calls occur.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit import shared_dictionary as codec
from synsteg_audit.budget_ledger import CampaignLedger

SCHEMA = "synsteg-recovery-development-audit-v1"
RECORD_FIELDS = ("record_id", "source_group_id", "split", "topic", "genre", "target_words", "prompt_sha256")
ATTEMPT_STATUSES = {"completed", "failed_or_unknown_no_retry", "reserved_outcome_unknown",
                    "accounting_bound_violation_stop"}
RECEIVER_PATH = ROOT / "scripts/run_shared_dictionary_pilot.py"
INPUT_KEYS = ["key_hex", "public_dictionary", "public_protocol", "stego_text"]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_fixture(kind, source_ids, n=32):
    """Domain-separated public reproducibility data, not confidential keys."""
    value = [SCHEMA, "PUBLIC-NONPRODUCTION-FIXTURE", kind, list(source_ids)]
    return hashlib.sha256(canonical(value)).digest()[:n]


def isolated_decode(stego_text, key, dictionary, protocol):
    request = {"key_hex": key.hex(), "stego_text": stego_text,
               "public_dictionary": dictionary.to_dict(), "public_protocol": protocol.to_dict()}
    # Preserve only OS startup essentials; never forward provider credentials.
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP")
                   if name in os.environ}
    environment.update(PATH=os.defpath, PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="synsteg-recovery-receiver-") as directory:
        child = subprocess.run([sys.executable, "-I", str(RECEIVER_PATH), "receiver"],
            input=canonical(request), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=directory, env=environment, timeout=60, check=False)
    try:
        reply = json.loads(child.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Receiver returned invalid JSON") from error
    if child.returncode != 0 or reply.get("status") != "decoded":
        raise ValueError("Receiver rejected frame")
    if (reply.get("input_keys") != INPUT_KEYS
            or reply.get("io_guard_self_checks") != {"file": True, "socket": True, "subprocess": True}
            or reply.get("public_dictionary_sha256") != dictionary.sha256
            or reply.get("public_protocol_sha256") != protocol.sha256):
        raise ValueError("Receiver isolation/protocol evidence mismatch")
    payload = bytes.fromhex(reply["payload_hex"])
    return payload, {**reply, "actual_returncode": child.returncode,
                     "stderr_sha256": hashlib.sha256(child.stderr).hexdigest()}


def snapshot(run):
    files = [run / "run.json", *sorted((run / "attempts").glob("*.json"))]
    return {path.relative_to(run).as_posix(): file_hash(path) for path in files}


def read_run(run):
    """Validate the immutable plan and retain unattempted planned records."""
    run = Path(run).resolve()
    before = snapshot(run)
    manifest = load_json(run / "run.json")
    if manifest.get("schema") != "synsteg-budget-generation-run-v1":
        raise ValueError("Unsupported generation run schema")
    if manifest.get("split") != "development":
        raise ValueError("This audit accepts development only; test/train/validation stay closed")
    if not all(isinstance(manifest.get(key), str) and manifest[key]
               for key in ("provider", "model", "prompts_sha256")):
        raise ValueError("Missing generation identity")
    records = manifest.get("selected_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Nonempty selected_records plan required")
    if manifest.get("selected_records_count") != len(records):
        raise ValueError("Planned count differs from selected_records")
    by_id = {}
    groups = set()
    for row in records:
        if not isinstance(row, dict) or any(key not in row for key in RECORD_FIELDS):
            raise ValueError("Incomplete planned record")
        if row["split"] != "development":
            raise ValueError("This audit accepts development records only")
        if not isinstance(row["record_id"], str) or not row["record_id"] or row["record_id"] in by_id:
            raise ValueError("Duplicate or invalid planned record_id")
        if not isinstance(row["source_group_id"], str) or not row["source_group_id"] or row["source_group_id"] in groups:
            raise ValueError("A development run must have distinct source groups")
        if type(row["target_words"]) is not int or row["target_words"] not in (250, 500):
            raise ValueError("Recovery-v1 target_words must be 250 or 500")
        by_id[row["record_id"]] = row
        groups.add(row["source_group_id"])
    attempts = {}
    for name in sorted(before):
        if not name.startswith("attempts/"):
            continue
        attempt = load_json(run / name)
        if attempt.get("schema") != "synsteg-budget-generation-attempt-v1":
            raise ValueError("Unsupported attempt schema")
        row = attempt.get("record", {})
        record_id = row.get("record_id")
        if record_id not in by_id or record_id in attempts:
            raise ValueError("Unexpected or duplicate attempt")
        if any(row.get(key) != by_id[record_id][key] for key in RECORD_FIELDS):
            raise ValueError("Attempt differs from frozen planned record")
        if any(attempt.get(key) != manifest[key] for key in ("provider", "model")):
            raise ValueError("Attempt provider/model differs from run")
        if attempt.get("status") not in ATTEMPT_STATUSES:
            raise ValueError("Unknown attempt status")
        attempts[record_id] = attempt
    if before != snapshot(run):
        raise ValueError("Generation inputs changed while reading")
    return manifest, attempts, before


def generation_outcome(attempt):
    if attempt is None:
        return "not_attempted", None, {}
    if attempt["status"] != "completed":
        return attempt["status"], None, {}
    result = attempt.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("cover_text"), str):
        raise ValueError("Completed attempt lacks a textual cover")
    metadata = {key: result.get(key) for key in ("stop_reason", "hit_output_limit", "usage", "seed")}
    if result.get("stop_reason") == "refusal" or result.get("refusal") is True:
        return "refusal", None, metadata
    text = result["cover_text"]
    if not text.strip():
        return "empty_cover", None, metadata
    capped = result.get("hit_output_limit") is True or result.get("stop_reason") in {"max_tokens", "length"}
    return "completed_at_output_limit" if capped else "completed", text, metadata


def roundtrip(cover, source_ids, dictionary, framing, bits, unavailable_reason=None):
    protocol = codec.Protocol(min_group_size=1, max_bits_per_slot=1, framing=framing)
    key = public_fixture("key", source_ids)
    payload = public_fixture("payload", source_ids, bits // 8)
    row = {"framing": framing, "selector": "canonical", "payload_bits": bits,
           "frame_overhead_bits": protocol.frame_overhead_bits, "protocol": protocol.to_dict(),
           "payload_hex": payload.hex(), "public_fixture_key_hex": key.hex(), "full_success": False}
    if cover is None:
        row.update(status="unavailable_cover", reason=unavailable_reason)
        return row
    row["capacity_bits"] = codec.analyze_text(cover, dictionary, protocol).capacity_bits
    try:
        encoded = codec.encode(cover, payload, key, dictionary, protocol, selector="canonical")
    except codec.CapacityError:
        row["status"] = "insufficient_capacity"
        return row
    try:
        decoded, receiver = isolated_decode(encoded.stego_text, key, dictionary, protocol)
    except (ValueError, subprocess.SubprocessError, OSError) as error:
        row.update(status="receiver_failure", error_type=type(error).__name__)
        return row
    row.update(status="decoded" if decoded == payload else "payload_mismatch", full_success=decoded == payload,
               decoded_payload_hex=decoded.hex(), stego_text=encoded.stego_text,
               stego_text_sha256=hashlib.sha256(encoded.stego_text.encode()).hexdigest(),
               receiver=receiver, encoding={k: v for k, v in asdict(encoded).items() if k != "stego_text"})
    return row


def summarize_arms(rows):
    counts = defaultdict(Counter)
    for row in rows:
        for name, result in row["arms"].items():
            counts[name]["planned"] += 1
            counts[name][result["status"]] += 1
            counts[name]["full_success"] += int(result["full_success"])
    return {name: {**dict(values), "full_success_fraction_all_planned": values["full_success"] / values["planned"]}
            for name, values in sorted(counts.items())}


def audit_run(manifest, attempts, dictionary):
    rows = []
    by_id = {}
    for index, planned in enumerate(manifest["selected_records"]):
        outcome, cover, metadata = generation_outcome(attempts.get(planned["record_id"]))
        row = {"plan_index": index, "record": planned, "generation_outcome": outcome,
               "generation_metadata": metadata, "cover_text": cover,
               "cover_words": len(cover.split()) if cover is not None else None,
               "cover_text_sha256": hashlib.sha256(cover.encode()).hexdigest() if cover is not None else None,
               "capacity_bits": codec.analyze_text(cover, dictionary,
                   codec.Protocol(min_group_size=1, max_bits_per_slot=1)).capacity_bits if cover is not None else None,
               "arms": {}}
        for framing in ("plain_length16", "aes_siv"):
            for bits in (8, 16):
                name = f"{framing}__canonical__{bits}"
                row["arms"][name] = roundtrip(cover, [planned["source_group_id"]], dictionary, framing, bits, outcome)
        rows.append(row)
        by_id[planned["record_id"]] = row
    bundles, leftovers = [], []
    for target in (250, 500):
        members = [row for row in manifest["selected_records"] if row["target_words"] == target]
        remainder = len(members) % 4
        complete_length = len(members) - remainder
        leftovers.extend(row["record_id"] for row in members[complete_length:])
        for start in range(0, complete_length, 4):
            group = members[start:start + 4]
            covers = [by_id[item["record_id"]]["cover_text"] for item in group]
            available = all(cover is not None for cover in covers)
            text = "\n\n".join(covers) if available else None
            ids = [item["source_group_id"] for item in group]
            row = {"bundle_id": digest([target, ids]), "target_words_per_member": target,
                   "record_ids": [item["record_id"] for item in group], "source_group_ids": ids,
                   "planned_members": 4, "available_members": sum(cover is not None for cover in covers),
                   "cover_text": text, "arms": {}}
            for bits in (8, 32, 128):
                row["arms"][f"aes_siv__canonical__{bits}"] = roundtrip(
                    text, ids, dictionary, "aes_siv", bits, "one_or_more_planned_members_unavailable")
            bundles.append(row)
    summary = {"provider": manifest["provider"], "model": manifest["model"],
               "planned_source_groups": len(rows), "generation_outcomes": dict(Counter(row["generation_outcome"] for row in rows)),
               "single_text_arms": summarize_arms(rows), "four_source_bundles": {
                   "planned_complete_bundles": len(bundles), "arms": summarize_arms(bundles),
                   "unbundled_tail_record_ids": leftovers,
                   "tail_policy": "Fewer than four remaining planned members are reported, not turned into a shorter bundle."}}
    return summary, rows, bundles


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def audit(runs, dictionary_path, output):
    paths = [Path(run).resolve() for run in runs]
    # The runner holds this same OS lock while mutating its plan/attempts.
    # Persistent lock files are not treated as evidence of an active process.
    with ExitStack() as stack:
        for path in sorted(set(paths)):
            stack.enter_context(CampaignLedger(path / "run_guard", "20").locked())
        return _audit_locked(paths, dictionary_path, output)


def _audit_locked(runs, dictionary_path, output):
    dictionary_path, output = Path(dictionary_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new audit directory")
    dictionary_bytes = dictionary_path.read_bytes()
    dictionary_hash = hashlib.sha256(dictionary_bytes).hexdigest()
    dictionary = codec.PublicDictionary.from_dict(json.loads(dictionary_bytes))
    inputs, evaluated, seen = [], [], set()
    for run in runs:
        path = Path(run).resolve()
        manifest, attempts, hashes = read_run(path)
        identity = (manifest["provider"], manifest["model"])
        if identity in seen:
            raise ValueError("Only one development run per provider/model is allowed")
        seen.add(identity)
        summary, rows, bundles = audit_run(manifest, attempts, dictionary)
        summary["input_manifest_sha256"] = hashes["run.json"]
        evaluated.append((summary, rows, bundles))
        inputs.append({"path": str(path), "hashes": hashes})
    if not evaluated:
        raise ValueError("At least one development run is required")
    for item in inputs:
        if snapshot(Path(item["path"])) != item["hashes"]:
            raise ValueError("Generation inputs changed during audit; no report written")
    if file_hash(dictionary_path) != dictionary_hash:
        raise ValueError("Public dictionary changed during audit")
    report = {"schema": SCHEMA, "scope": "development_only_new_recovery_baseline",
              "public_dictionary_sha256": dictionary.sha256, "dictionary_file_sha256": dictionary_hash,
              "audit_script_sha256": file_hash(__file__), "receiver_script_sha256": file_hash(RECEIVER_PATH),
              "codec_script_sha256": file_hash(ROOT / "src/synsteg_audit/shared_dictionary.py"),
              "input_snapshots": inputs, "models": [item[0] for item in evaluated],
              "public_fixture_keys_are_not_confidential": True,
              "receiver_scope": "Fresh isolated Python process and tested Python audit-hook I/O guard; not an OS sandbox.",
              "quality_metrics": {"status": "not_computed", "reason": "No independent language-model scorer or human ratings in this diagnostic."},
              "steganalysis": {"status": "not_computed"}, "test_split_opened": False,
              "limitations": ["WordNet classes do not guarantee contextual synonymy or meaning preservation.",
                  "Truncated nonempty covers remain in the planned denominator and codec checks, with their output-limit flag.",
                  "Plain framing is unauthenticated; successful roundtrips do not establish confidentiality or undetectability.",
                  "No independent model comparisons or inferential claims are made from these development data."]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    # Preserve the original source/license/provenance metadata byte-for-byte;
    # the receiver's canonical dictionary projection intentionally omits it.
    (output / "public_dictionary.json").write_bytes(dictionary_bytes)
    for index, (summary, rows, bundles) in enumerate(evaluated):
        name = f"model-{index + 1:02d}"
        (output / f"{name}-records.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        (output / f"{name}-bundles.jsonl").write_bytes(b"".join(canonical(row) + b"\n" for row in bundles))
        summary["records_file"] = f"{name}-records.jsonl"
        summary["bundles_file"] = f"{name}-bundles.jsonl"
    write_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="Repeat for each development model run")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data/shared_dictionary_recovery_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    args = parser.parse_args(argv)
    report = audit(args.run, args.dictionary, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "models": len(report["models"]), "scope": report["scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
