#!/usr/bin/env python3
"""Metadata-only first80 activation from root-verified immutable batch40.

Does not enable execution, calculate capacity, encode, decode, or call a model.
The coordinator must independently verify the actual Git origin of baseline_root.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
ID = "prospective-development-transport-first80-v1"
CP = "results/large-campaign-development-v2/batches/batch40/after-checkpoint.json"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable(path):
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "regular nonsymlink file required")
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) and len(raw) == after.st_size, "source changed during read")
    return raw


def write_once(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-primary-commit", required=True)
    args = parser.parse_args()
    root, baseline = args.root.resolve(), args.baseline_root.resolve()
    need(re.fullmatch("[0-9a-f]{40}", args.baseline_primary_commit) is not None, "exact coordinator Git commit required")
    activation_path = root / "experiments" / ID / "activation.json"
    need(not activation_path.exists() and not (root / "results" / ID).exists(), "first80 activation/output already exists")
    cp_raw = stable(baseline / CP)
    cp = json.loads(cp_raw)
    need(cp["status"] == "verified" and cp["completed_generations"] == cp["attempted"] == 80,
         "verified exact baseline80 required")
    need(cp["counts"] == {"completed": 80, "not_attempted": 220}
         and cp["model_counts"] == {"qwen05": 0, "tinyllama11": 80, "falcon3_1b": 0}
         and cp["planned_model_record_pairs"] == 300 and len(cp["records"]) == 300
         and not cp["closed_test_opened"] and cp["paid_calls"] == 0
         and not cp["consistency_issues"], "baseline cohort/scope differs")
    need(len(cp["run_files_sha256"]) == 86, "expected80 attempts+6run metadata files")
    snapshot = Path(CP).parent / "after"
    files = {CP: cp_raw}
    for name, digest in cp["run_files_sha256"].items():
        rel = Path(name)
        need(not rel.is_absolute() and ".." not in rel.parts, "unsafe snapshot path")
        data = stable(baseline / snapshot / rel)
        need(sha(data) == digest, "snapshot hash differs: " + name)
        files[(snapshot / rel).as_posix()] = data
    files[(snapshot / "files.json").as_posix()] = stable(baseline / snapshot / "files.json")
    need(json.loads(files[(snapshot / "files.json").as_posix()]) == cp["run_files_sha256"], "snapshot inventory differs")
    for name, raw in files.items():
        existing = root / name
        if existing.exists():
            need(stable(existing) == raw, "existing snapshot differs: " + name)
    for name, raw in files.items():
        if not (root / name).exists():
            write_once(root / name, raw)
    path = root / "scripts/run_prospective_development_transport_first80_v1.py"
    spec = importlib.util.spec_from_file_location("first80_preparation_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parent = runner.parent(root)
    dependencies = parent.dependencies(root, root / "runtime/prospective-development-transport-v1/frozen")
    runner.prior_inputs(parent, root / runner.PRIOR)
    runner.verify_prior_audit(parent, root)
    hashes = {name: sha(stable(root / name)) for name in runner.IMPLEMENTATION_FILES}
    activation = {
        "schema": "prospective-development-transport-first80-activation-v1", "experiment_id": ID,
        "execution_enabled": False, "status": "disabled_metadata_preparation_pending_root_git_gate",
        "feasibility_protocol_sha256": parent.PLAN_SHA,
        "feasibility_protocol_is_unchanged_and_execution_disabled": True,
        "activation_scope": "Only allnewTinyLlama61–80 publicconditions; exactprior1440reuse; nohistoricalreceiver/model/candidatecalls.",
        "observed_completed_covers_at_implementation_freeze": 80,
        "prior_information": {"first60_transport_results_known": True, "prior_public_encoder_attempts": 1440,
            "prior_exact_deliveries": 150, "prior_capacity_refusals": 1290, "prior_zero_capacity_conditions": 300,
            "ordinary_covers_completed_before_transport_input_freeze": 80,
            "not_pure_pre_data_preregistration": True, "no_new_data_claim_for_reused_rows": True},
        "prior": {"milestone": runner.PRIOR, "output_manifest_sha256": runner.PRIOR_MANIFEST_SHA,
            "archive_sha256": runner.PRIOR_ARCHIVE_SHA, "source_commit": "b8bc59d8646d1a0a8c85bd08dee62107413f2f27",
            "independent_audit": runner.PRIOR_AUDIT, "independent_audit_sha256": runner.PRIOR_AUDIT_SHA,
            "cumulative_receiver_jobs": 150, "new_only_recipient_jobs_not_to_use_as_prior": 24},
        "target": {"milestone_id": "tinyllama-first80", "batch": "batch40", "checkpoint": CP,
            "output": f"results/{ID}/milestones/tinyllama-first80", "completed_covers": 80,
            "cover_counts": {"completed": 80, "not_attempted": 220},
            "model_counts": {"qwen05": 0, "tinyllama11": 80, "falcon3_1b": 0},
            "full_model_record_denominator": 300, "total_cells": 15300, "primary_cells": 8100,
            "secondary16_cells": 7200, "public_cells": 7200, "original_fixed_resource_blocked_cells": 8100,
            "cumulative_available_public_conditions": 1920, "planned_new_public_conditions": 480,
            "reused_public_conditions": 1440, "new_source_orders": list(range(61, 81)),
            "selection": "Allfixedfirst80 TinyLlama; only61–80 newencode; noqualityselection.",
            "prior_receiver_replay": False, "input_binding": runner.BINDING,
            "unchanged_cell_rows": 14280, "updated_cell_rows": 1020, "planned_cells_per_record": 51,
            "missing_cover_cells": 11220, "available_resource_blocked_cells": 2160},
        "runtime": parent.runtime_identity(), "implementation_files_sha256": hashes,
        "inherited_implementation_files_sha256": {runner.PARENT_RUNNER: runner.PARENT_RUNNER_SHA,
                                                   runner.PARENT_AUDITOR: runner.PARENT_AUDITOR_SHA},
        "input_binding_files_sha256": None, "checkpoint_sha256": sha(cp_raw),
        "baseline_primary_commit": args.baseline_primary_commit, "new_transport_outcomes_known": False,
        "coordinator_actual_git_baseline_verified_by_helper": False,
        "git_origin_limitation": "Coordinator verifies remoteorigin; helperchecks suppliedimmutablebytes and recordscommit, notnetworkorigin.",
    }
    runner.snapshot_records(parent, root, dependencies[1], activation)
    write_once(activation_path, (json.dumps(activation, ensure_ascii=False, indent=2) + "\n").encode())
    receipt = {"schema": "prospective-first80-disabled-activation-preparation-v1",
        "status": "disabled_activation_created", "copied_snapshot_file_count": len(files),
        "copied_files_sha256": {name: sha(raw) for name, raw in files.items()},
        "activation_sha256": sha(stable(activation_path)), "baseline_primary_commit": args.baseline_primary_commit,
        "model_calls": 0, "encoder_calls": 0, "decoder_calls": 0, "capacity_checks": 0,
        "baseline_root_supplied_by_coordinator": str(baseline)}
    receipt_path = root / "results" / (ID + "-preparation") / "disabled-activation-preparation.json"
    write_once(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({"status": receipt["status"], "copied_snapshot_files": len(files), "encoder_calls": 0}))


if __name__ == "__main__":
    main()
