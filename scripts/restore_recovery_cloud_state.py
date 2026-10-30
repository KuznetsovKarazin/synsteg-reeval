#!/usr/bin/env python3
"""Verify/restore complete Git-stored recovery state without rewriting evidence.

Never resume a standalone/API run from attempts alone: the frozen runner creates
a new ledger when ledger.json is absent, which can repeat an already completed
generation. This helper rejects that partial-state condition before restoration.
No network, generation, credential/environment lookup, or frozen-file edits.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
KEY_PATTERN = re.compile(rb"(?<![A-Za-z0-9])sk-(?:ant|or)-[A-Za-z0-9_-]{20,}")
TERMINAL = {"completed", "failed_or_unknown_no_retry", "accounting_bound_violation_stop"}
PENDING = "reserved_outcome_unknown"


class RestoreError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode())


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RestoreError("duplicate_json_key")
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value
            or PurePosixPath(value).is_absolute() or str(PurePosixPath(value)) != value):
        raise RestoreError("unsafe_relative_path")
    for part in value.split("/"):
        if part in ("", ".", "..") or part.endswith((" ", ".")):
            raise RestoreError("unsafe_relative_path")
        if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]):
            raise RestoreError("nonportable_reserved_path")
    return value


def safe_path(repo, name):
    path = repo / relative(name)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise RestoreError("symlink_path_refused")
        if part == repo:
            break
    return path


def inventory_map(entries):
    if not isinstance(entries, list) or not entries:
        raise RestoreError("empty_or_invalid_snapshot_inventory")
    found, folded = {}, set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RestoreError("invalid_snapshot_inventory_entry")
        name = relative(entry.get("path"))
        folded_name = unicodedata.normalize("NFC", name).casefold()
        if name in found or folded_name in folded:
            raise RestoreError("snapshot_path_collision")
        if name.endswith(".lock") or PurePosixPath(name).name == ".writer.lock":
            raise RestoreError("lock_file_in_snapshot")
        if not valid_hash(entry.get("sha256")) or type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
            raise RestoreError("invalid_snapshot_file_identity")
        found[name] = entry
        folded.add(folded_name)
    for name in found:
        if any(unicodedata.normalize("NFC", str(parent)).casefold() in folded
               for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise RestoreError("snapshot_file_directory_collision")
    return found


def verify_tree(root, files):
    if not root.is_dir() or root.is_symlink():
        raise RestoreError("snapshot_directory_missing_or_unsafe")
    actual = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RestoreError("symlink_in_snapshot")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RestoreError("nonregular_snapshot_file")
        name = path.relative_to(root).as_posix()
        relative(name)
        if path.name.endswith(".lock") or path.name == ".writer.lock":
            raise RestoreError("lock_file_in_snapshot")
        actual.add(name)
    if actual != set(files):
        raise RestoreError("snapshot_inventory_differs_from_tree")
    for name, entry in files.items():
        path = root / name
        if path.stat().st_size != entry["size_bytes"]:
            raise RestoreError("snapshot_file_size_mismatch")
        data = path.read_bytes()
        if digest(data) != entry["sha256"]:
            raise RestoreError("snapshot_file_hash_mismatch")
        if KEY_PATTERN.search(data):
            raise RestoreError("provider_credential_pattern_in_snapshot")


def current_runtime():
    try:
        packages = {name: importlib.metadata.version(name) for name in
                    ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")}
    except importlib.metadata.PackageNotFoundError:
        return None
    return {"schema": "synsteg-local-runtime-recovery-v1", "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "packages": packages}


def verify_code(repo, plan):
    bindings = {"scripts/run_budget_generation.py": plan.get("runner_sha256"),
                "src/synsteg_audit/budget_ledger.py": plan.get("ledger_code_sha256")}
    transport = plan.get("transport_code_sha256")
    if plan["provider"] == "claude":
        if not isinstance(transport, dict) or set(transport) != {"run_claude_smoke.py", "check_claude_access.py"}:
            raise RestoreError("invalid_frozen_transport_identity")
        bindings.update({"scripts/" + name: value for name, value in transport.items()})
    for name, expected in bindings.items():
        path = safe_path(repo, name)
        if not valid_hash(expected) or not path.is_file() or digest(path.read_bytes()) != expected:
            raise RestoreError("frozen_implementation_differs_from_repository")


def verify_state(repo, source, snapshot_type, runtime_reader=None):
    plans = sorted((source / "models").glob("*/run.json")) if snapshot_type == "campaign" else [source / "run.json"]
    if not plans or any(not path.is_file() for path in plans):
        raise RestoreError("snapshot_has_no_run_plan")
    if snapshot_type == "campaign":
        assets = read_json(source / "assets.json")
        if assets.get("schema") != "synsteg-development-assets-v1" or assets.get("test_execution_enabled") is not False:
            raise RestoreError("invalid_campaign_assets")
        for name, expected in assets.get("files", {}).items():
            path = safe_path(repo, name)
            if not valid_hash(expected) or not path.is_file() or digest(path.read_bytes()) != expected:
                raise RestoreError("frozen_campaign_assets_differ_from_repository")
        if digest((source / "development-prompts.jsonl").read_bytes()) != assets.get("development_prompts_sha256"):
            raise RestoreError("campaign_prompts_hash_mismatch")
        plan_hashes = {path.parent.name: digest(path.read_bytes()) for path in plans}
        if assets.get("model_plan_hashes") != plan_hashes:
            raise RestoreError("campaign_model_plan_inventory_mismatch")
    active_runtime = (runtime_reader or current_runtime)()
    ledger_entries, ledger_caps, expected_entries, reports = {}, {}, {}, []
    pending = []
    for plan_path in plans:
        run = plan_path.parent
        plan = read_json(plan_path)
        if (plan.get("schema") != "synsteg-budget-generation-run-v1" or plan.get("provider") not in ("local", "claude")
                or plan.get("split") != "development"):
            raise RestoreError("only_frozen_development_run_plans_are_supported")
        verify_code(repo, plan)
        records = plan.get("selected_records")
        if not isinstance(records, list) or len(records) != plan.get("selected_records_count") or not records:
            raise RestoreError("invalid_planned_denominator")
        record_map = {}
        for record in records:
            if not isinstance(record, dict) or record.get("split") != "development" or record.get("record_id") in record_map:
                raise RestoreError("invalid_or_duplicate_planned_record")
            record_map[record["record_id"]] = record
        if snapshot_type == "campaign" and plan.get("prompts_sha256") != assets["development_prompts_sha256"]:
            raise RestoreError("run_uses_different_development_prompts")
        attempts = sorted((run / "attempts").glob("*.json"))
        if any(path.is_file() and path not in attempts for path in (run / "attempts").rglob("*")):
            raise RestoreError("unexpected_attempt_member")
        summary_path = run / "summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else {}
        ledger_path = run / "local_ledger/ledger.json" if plan["provider"] == "local" else source / "ledger/ledger.json"
        if snapshot_type == "standalone" and plan["provider"] == "claude":
            raise RestoreError("paid_standalone_snapshot_requires_complete_shared_campaign")
        summary_has_attempts = (summary.get("completed", 0) > 0
                               or summary.get("requests_attempted_this_invocation", 0) > 0
                               or bool(summary.get("attempts")))
        if (attempts or summary_has_attempts) and not ledger_path.is_file():
            raise RestoreError("missing_ledger_would_allow_duplicate_run")
        runtime = read_json(run / "runtime.json") if (run / "runtime.json").is_file() else None
        if plan["provider"] == "local" and attempts and runtime is None:
            raise RestoreError("missing_runtime_for_attempted_local_run")
        runtime_match = runtime is None or runtime == active_runtime
        if ledger_path.is_file() and ledger_path not in ledger_entries:
            ledger = read_json(ledger_path)
            if (ledger.get("schema") != "synsteg-budget-ledger-recovery-v1"
                    or type(ledger.get("budget_nanodollars")) is not int
                    or not 0 < ledger["budget_nanodollars"] <= 20_000_000_000
                    or not isinstance(ledger.get("attempts"), dict)):
                raise RestoreError("invalid_ledger_identity")
            ledger_entries[ledger_path] = ledger["attempts"]
            ledger_caps[ledger_path] = ledger["budget_nanodollars"]
            expected_entries[ledger_path] = set()
            if ledger.get("blocked_reason"):
                pending.append("ledger_blocked_after_bound_violation")
        entries = ledger_entries.get(ledger_path, {})
        if entries and plan["provider"] == "local" and runtime is None:
            raise RestoreError("missing_runtime_for_attempted_local_run")
        if ledger_path in ledger_caps:
            try:
                plan_cap = Decimal(str(plan["budget_usd"])) * 1_000_000_000
            except (KeyError, InvalidOperation):
                raise RestoreError("invalid_plan_budget") from None
            if plan_cap != ledger_caps[ledger_path]:
                raise RestoreError("plan_and_ledger_budget_differ")
        if any(item.get("identity") not in entries for item in summary.get("attempts", [])):
            raise RestoreError("missing_ledger_entry_would_allow_duplicate_run")
        seen = set()
        for attempt_path in attempts:
            attempt = read_json(attempt_path)
            identity = attempt.get("identity")
            record = attempt.get("record", {})
            rid = record.get("record_id")
            if (attempt.get("schema") != "synsteg-budget-generation-attempt-v1" or not valid_hash(identity)
                    or attempt_path.name != identity + ".json" or identity in seen
                    or attempt.get("provider") != plan["provider"] or attempt.get("model") != plan["model"]
                    or rid not in record_map or record != record_map[rid]
                    or identity != canonical_hash([plan["provider"], plan["model"], rid])):
                raise RestoreError("attempt_identity_or_plan_mismatch")
            seen.add(identity)
            ledger = entries.get(identity)
            if ledger is None:
                raise RestoreError("missing_ledger_entry_would_allow_duplicate_run")
            if (ledger.get("identity") != identity or ledger.get("metadata") !=
                    {"record": record, "provider": plan["provider"], "model": plan["model"]}
                    or not valid_hash(attempt.get("request_fingerprint"))
                    or ledger.get("request_fingerprint") != attempt["request_fingerprint"]):
                raise RestoreError("attempt_and_ledger_identity_differ")
            fingerprint = canonical_hash({"record": record, "provider": plan["provider"], "model": plan["model"],
                "revision": plan["revision"], "dtype": plan["dtype"], "settings": plan["settings"],
                "pricing_sha256": plan["pricing_sha256"], "prompts_sha256": plan["prompts_sha256"],
                "runtime_sha256": canonical_hash(runtime) if plan["provider"] == "local" else None,
                "implementation": {name: plan[name] for name in
                    ("runner_sha256", "ledger_code_sha256", "transport_code_sha256")}})
            if attempt["request_fingerprint"] != fingerprint:
                raise RestoreError("request_fingerprint_does_not_match_frozen_inputs")
            state, attempt_state = ledger.get("status"), attempt.get("status")
            if state not in TERMINAL | {PENDING} or attempt_state not in TERMINAL | {PENDING}:
                raise RestoreError("unknown_attempt_status")
            if attempt_state != state and not (attempt_state == PENDING and state in TERMINAL):
                raise RestoreError("attempt_and_ledger_status_conflict")
            if state == "completed" and attempt_state == "completed" and attempt.get("result") != ledger.get("result"):
                raise RestoreError("attempt_and_ledger_result_conflict")
            if state != "completed":
                pending.append(identity)
            expected_entries[ledger_path].add(identity)
        # A crash after ledger reservation but before the attempt file is safe
        # to preserve, but it remains blocked. Never synthesize a completed call.
        for identity, entry in entries.items():
            meta = entry.get("metadata", {})
            rid = meta.get("record", {}).get("record_id")
            if meta.get("provider") != plan["provider"] or meta.get("model") != plan["model"]:
                continue
            if rid in record_map and identity not in seen and entry.get("status") == PENDING:
                if identity != canonical_hash([plan["provider"], plan["model"], rid]):
                    raise RestoreError("unmatched_pending_reservation_identity")
                expected_entries[ledger_path].add(identity)
                pending.append(identity)
        reports.append({"run": run.relative_to(source).as_posix(), "provider": plan["provider"],
                        "model": plan["model"], "planned": len(records), "attempt_files": len(attempts),
                        "runtime_matches_current_process": runtime_match,
                        "runtime_record_present": runtime is not None})
    for path, entries in ledger_entries.items():
        if set(entries) != expected_entries[path]:
            raise RestoreError("ledger_contains_attempts_missing_from_snapshot")
        for entry in entries.values():
            if any(type(entry.get(k)) is not int or entry[k] < 0 for k in ("reserved_nanodollars", "accounted_nanodollars")):
                raise RestoreError("invalid_ledger_amount")
            if entry["status"] == "completed" and entry["accounted_nanodollars"] > entry["reserved_nanodollars"]:
                raise RestoreError("completed_cost_exceeds_reservation")
        if sum(entry["accounted_nanodollars"] for entry in entries.values()) > ledger_caps[path]:
            pending.append("ledger_accounting_exceeds_campaign_cap")
    return {"runs": reports, "unresolved_or_failed_attempts": sorted(set(pending)),
            "resume_eligible_in_current_runtime": not pending and all(r["runtime_matches_current_process"] for r in reports),
            "new_requests_authorized_by_verification": False,
            "absolute_evidence_paths_rewritten": False,
            "other_required_checks": ["verify task authorization and remaining work-window before new execution",
                                      "frozen prompts/dictionary for standalone runs", "pinned model cache hashes"]}


def operate(manifest_path, *, repo=ROOT, restore=False, runtime_reader=None):
    repo = Path(repo).absolute()
    if any(path.is_symlink() for path in (repo, *repo.parents)):
        raise RestoreError("symlink_repository_root")
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    try:
        manifest_name = manifest_path.relative_to(repo).as_posix()
    except ValueError:
        raise RestoreError("manifest_must_be_inside_repository") from None
    manifest_path = safe_path(repo, manifest_name)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest, object_pairs_hook=unique_object)
    if manifest.get("schema") != "synsteg-cloud-snapshot-v1" or manifest.get("type") not in ("campaign", "standalone"):
        raise RestoreError("unsupported_cloud_snapshot_manifest")
    source = safe_path(repo, manifest.get("snapshot_root"))
    target = safe_path(repo, manifest.get("suggested_target"))
    if source == target or source in target.parents or target in source.parents:
        raise RestoreError("snapshot_and_target_paths_overlap")
    files = inventory_map(manifest.get("files"))
    verify_tree(source, files)
    diagnostic = verify_state(repo, source, manifest["type"], runtime_reader)
    status = "verified_no_files_written"
    if restore:
        if os.path.lexists(target):
            verify_tree(target, files)
            status = "already_restored_identical"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".cloud-restore-", dir=target.parent) as temp:
                staged = Path(temp) / "state"
                staged.mkdir()
                for name, entry in files.items():
                    data = (source / name).read_bytes()
                    if len(data) != entry["size_bytes"] or digest(data) != entry["sha256"]:
                        raise RestoreError("source_changed_during_restore")
                    destination = staged / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                verify_tree(source, files)
                verify_tree(staged, files)
                if manifest_path.read_bytes() != raw_manifest:
                    raise RestoreError("manifest_changed_during_restore")
                target.mkdir(mode=0o700, exist_ok=False)
                try:
                    for child in staged.iterdir():
                        shutil.move(str(child), str(target / child.name))
                    verify_tree(target, files)
                except BaseException:
                    shutil.rmtree(target)
                    raise
            status = "restored_verified_state"
    return {"schema": "synsteg-cloud-restore-verification-v1", "status": status,
            "manifest_sha256": digest(raw_manifest), "snapshot_root": manifest["snapshot_root"],
            "suggested_target": manifest["suggested_target"], "files": len(files),
            "created_from_commit": manifest.get("created_from_commit"),
            "commit_provenance_independently_verified": False, **diagnostic}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "restore"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        report = operate(args.manifest, repo=args.repo, restore=args.action == "restore")
    except (RestoreError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "refused", "diagnostic": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
