#!/usr/bin/env python3
"""Build bounded activation only from a coordinator's ACTUAL Git readback.

This helper does no Git fetch, download, native process, tokenizer or model call.
It cannot attest that the coordinator performed the Git read: that external
action must precede the supplied exact readback receipt. It writes immutable new
activation/evidence files and validates both receiver and parent paths read-only.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from synsteg_audit import framed_interval_bounded_v1_activation as act
from synsteg_audit.framed_interval_llm_v1 import canonical_json
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json

OUTPUT = "results/framed-interval-bounded-v1/native-pilot-v1"
PREPARATION = "results/framed-interval-bounded-v1-preparation"


def prepare(root, readback_sha):
    root = Path(root).resolve()
    proof = json.loads(act.checked_bytes(root, {"path": act.READBACK_PATH, "sha256": readback_sha}))
    if (set(proof) != {"schema", "commit", "all_files_exact", "files_sha256"}
            or proof["schema"] != act.VERSION+"-source-readback-v1"
            or proof["all_files_exact"] is not True
            or set(proof["files_sha256"]) != act.REQUIRED_SOURCES):
        raise ValueError("exact complete external Git readback required")
    act.prior._hex(proof["commit"], 40)
    for path, sha in proof["files_sha256"].items():
        act.checked_bytes(root, {"path": path, "sha256": sha})
        if path in act.PINNED and sha != act.PINNED[path]:
            raise ValueError("pinned source identity differs")
    if (root/OUTPUT).exists():
        raise FileExistsError("campaign output already exists; no retry")
    freeze = {"schema": act.VERSION+"-source-freeze-v1",
        "commit": proof["commit"], "files_sha256": proof["files_sha256"]}
    atomic_json(root/act.FREEZE_PATH, freeze)
    descriptor = lambda path: {"path": path, "sha256": act.sha_file(root/path)}
    protocol = json.loads((root/act.PUBLIC_PATH).read_bytes())
    activation = {"schema": act.VERSION+"-activation-v1", "experiment_id": act.VERSION,
        "enabled": True, "final_freeze_commit": proof["commit"],
        "freeze_manifest": descriptor(act.FREEZE_PATH),
        "source_readback": {"path": act.READBACK_PATH, "sha256": readback_sha},
        "public_protocol": descriptor(act.PUBLIC_PATH),
        "sender_fixtures": descriptor(act.FIXTURES_PATH),
        "expected_public_runs_sha256": act.digest(canonical_json(protocol["runs"])),
        "prior_runtime_activation": {"path": act.PRIOR_ACTIVATION_PATH, "sha256": act.PRIOR_ACTIVATION_SHA}}
    atomic_json(root/act.ACTIVATION_PATH, activation)
    activation_sha = act.sha_file(root/act.ACTIVATION_PATH)
    validated = {}
    for scope in (True, False):
        loaded = act.load_activation(root, act.ACTIVATION_PATH, activation_sha, act.INTENT,
            receiver_scope=scope)
        validated["receiver" if scope else "coordinator"] = {
            "passed": True, "verified_resource_files": loaded.resource_report["verified_files"],
            "prompt_lengths": {name: len(ids) for name, ids in loaded.preflight["prompt_ids"].items()},
            "private_fixture_read_required": not scope}
    receipt = {"schema": act.VERSION+"-activation-validation-v1", "status": "passed",
        "activation_sha256": activation_sha, "source_readback_sha256": readback_sha,
        "source_freeze_commit": proof["commit"], "scopes": validated,
        "native_calls": 0, "model_calls": 0, "tokenizer_calls": 0,
        "scope": "Fresh file/runtime/resource checks reuse saved tokenizer evidence; no repeat of preflight/pilot",
        "requires_activation_and_intent_Git_commit_and_actual_readback_before_launch": True}
    atomic_json(root/PREPARATION/"activation-validation.json", receipt)
    intent = {"schema": act.VERSION+"-execution-intent-v1", "state": "prepared_not_started",
        "source_freeze_commit": proof["commit"], "activation": descriptor(act.ACTIVATION_PATH),
        "source_readback": activation["source_readback"], "output": OUTPUT,
        "planned_generations": 16, "planned_fresh_stego_receivers": 8,
        "max_generation_tokens": 12288, "max_receiver_token_evaluations_excluding_prompts": 6144,
        "attempts_per_endpoint": 1, "retry_permitted": False, "unknown_STEP_retried": False,
        "paid_calls": 0, "closed_test_accessed": False,
        "authorization": "Existing user authorization for free cloud continuation; no new permission requested",
        "prelaunch_gate": "Coordinator must commit and actually reread this intent and activation/evidence before launch"}
    atomic_json(root/PREPARATION/"execution-intent.json", intent)
    return {"status": "prepared_and_validated_no_model_calls", "activation_sha256": activation_sha,
        "must_save_and_readback_before_launch": [
            act.READBACK_PATH, act.FREEZE_PATH, act.ACTIVATION_PATH,
            PREPARATION+"/activation-validation.json", PREPARATION+"/execution-intent.json"],
        "command_after_Git_readback": [
            sys.executable, "experiments/framed-interval-bounded-v1/run_bounded.py", "run",
            "--activation", act.ACTIVATION_PATH, "--activation-sha256", activation_sha,
            "--intent", act.INTENT, "--output", OUTPUT]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-readback-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.source_readback_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
