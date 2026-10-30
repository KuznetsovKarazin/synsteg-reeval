#!/usr/bin/env python3
"""Build only a DISABLED four-case activation and exact future freeze map.

No native process, tokenizer, compiler, downloader or model is called. Run with
the same Python environment as the future launcher. Before preflight, omit its
arguments; afterwards pass the actual receipt path and its trusted SHA256 into
a NEW output directory. The coordinator must still freeze/read back the mapped
files, create the final freeze manifest and separate enabled activation, then
call the unchanged load_activation gate before any production launch.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synsteg_audit.framed_interval_llm_v1 import canonical_json
from synsteg_audit.framed_interval_llm_v1_activation import (
    EXECUTION_PATH, EXECUTION_SHA, FROZEN_COMPONENTS, PROPOSAL_PATH, PROPOSAL_SHA,
    PUBLIC_PATH, PUBLIC_SHA, REQUIRED_SOURCES, checked_bytes,
    digest, safe_path, sha_file, verify_resource_files,
)
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json
from synsteg_audit.framed_interval_llm_v1_orchestration import mkdir_durable
from synsteg_audit.framed_interval_llm_v1_preflight import validate_preflight_receipt
from synsteg_audit.framed_interval_llm_v1_qwen import (
    CPU_SHA256, MODEL_SHA256, PARAMETERS, RUNTIME_COMMIT,
)
from synsteg_audit.framed_interval_llm_v1_vocab_collect import PLAN_SHA256, SOURCE_PATHS
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import current_runtime

SELF = "scripts/prepare_framed_interval_llm_run_activation.py"
BUILD_PLAN = {"path": "experiments/framed-interval-llm-v1/vocab-build-plan-v1.json",
    "sha256": "667850b012f6e7c06487a1fd763c2b82146da332256cf74f6dcdc5cb21e6a084"}
PLAN_PATH = "experiments/framed-interval-llm-v1/tokenizer-preflight-proposal.json"


def descriptor(root, name):
    return {"path": name, "sha256": sha_file(safe_path(root, name))}


def prepare(root, output, *, preflight_path=None, preflight_sha=None):
    root = Path(root).resolve()
    if (preflight_path is None) != (preflight_sha is None):
        raise ValueError("actual preflight path and trusted SHA must be supplied together")
    destination = safe_path(root, output)
    if destination.exists(): raise FileExistsError("preparation output already exists; use a new path")
    build = json.loads(checked_bytes(root, BUILD_PLAN))
    for name, expected in {PROPOSAL_PATH: PROPOSAL_SHA, PUBLIC_PATH: PUBLIC_SHA,
            EXECUTION_PATH: EXECUTION_SHA, **FROZEN_COMPONENTS,
            **build["sources_sha256"]}.items():
        checked_bytes(root, {"path": name, "sha256": expected})
    source_bundle = digest(canonical_json({name: sha_file(safe_path(root, name)) for name in SOURCE_PATHS}))
    if source_bundle != build["source_bundle_sha256"]:
        raise ValueError("frozen collector dependency bundle differs")
    resources = verify_resource_files(root)  # fresh byte check; allow_download=False
    resource_entry = build["resource_restore_receipt"]
    recorded = json.loads(checked_bytes(root, resource_entry))
    fields = ("status", "expected_files", "verified_files", "verified_file_identities",
              "frozen_resource_identity_sha256", "bridge_executable_verified", "manifest_sha256")
    if any(recorded.get(name) != resources.get(name) for name in fields):
        raise ValueError("restoration receipt and current41 resources differ")
    runtime = current_runtime()
    historical = json.loads(safe_path(root, EXECUTION_PATH).read_bytes())["channel_resources"]
    for name in ("python", "platform", "machine", "libc"):
        if runtime[name] != historical[name]: raise ValueError("historical runtime metadata differs: " + name)
    system_entry = build["system_library_manifest"]
    systems = json.loads(checked_bytes(root, system_entry))["files_sha256"]
    if not systems: raise ValueError("system-library identities required")
    for path, expected in systems.items():
        if not Path(path).is_absolute() or sha_file(path) != expected:
            raise ValueError("current system library differs: " + path)
    protocol = json.loads(checked_bytes(root, {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA}))
    plan_entry = {"path": PLAN_PATH, "sha256": PLAN_SHA256}
    plan_bytes = checked_bytes(root, plan_entry)
    libraries = {name: row["sha256"] for name, row in resources["verified_file_identities"].items()
                 if name.startswith("runtime/")}
    obj = {"schema": "framed-interval-llm-v1-activation-v1", "experiment_id": "framed-interval-llm-v1",
        "enabled": False, "final_freeze_commit": None, "coordinator_readback": None,
        "freeze_manifest": None, "public_protocol": {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA},
        "expected_public_runs_sha256": digest(canonical_json(protocol["runs"])),
        "resource_receipt": resource_entry, "runtime": runtime,
        "qwen_binding_manifest": {"model_sha256": MODEL_SHA256, "runtime_commit": RUNTIME_COMMIT,
            "cpu_backend_sha256": CPU_SHA256,
            "bridge_sha256": resources["verified_file_identities"]["seeded_pair_llama_bridge"]["sha256"],
            "shared_libraries_sha256": libraries, "python_runtime_sha256": digest(canonical_json(runtime)),
            "quantizer_sha256": sha_file(safe_path(root, "src/synsteg_audit/seeded_pair_llm.py")),
            "special_ids_sha256": None, "vocabulary_size": None, "masked_tokens": None,
            "parameters": dict(PARAMETERS)},
        "ready": None, "preflight_plan": plan_entry, "preflight_receipt": None,
        "preflight_inventory": None, "collector_source_sha256": source_bundle,
        "native_system_libraries_sha256": systems}
    dependencies = set(REQUIRED_SOURCES) | set(SOURCE_PATHS) | {
        SELF, EXECUTION_PATH, PLAN_PATH, BUILD_PLAN["path"], resource_entry["path"], system_entry["path"],
        "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py",
        "experiments/framed-interval-llm-v1/run_fixed512_runtime_compat_v1.py",
        "experiments/framed-interval-llm-v1/vocab_runtime_compat_v1.py"}
    missing = ["actual_vocab_only_preflight_receipt_and_inventory"]
    if preflight_path is not None:
        entry = {"path": preflight_path, "sha256": preflight_sha}
        receipt = json.loads(checked_bytes(root, entry))
        inventory = {"path": receipt["inventory"]["path"], "sha256": receipt["inventory"]["sha256"]}
        checked = validate_preflight_receipt(receipt, plan_bytes=plan_bytes,
            inventory_bytes=checked_bytes(root, inventory), expected_plan_sha256=PLAN_SHA256,
            expected_resource_identity_sha256=resources["frozen_resource_identity_sha256"],
            expected_native_libraries=libraries, expected_collector_source_sha256=source_bundle,
            expected_ready=receipt["ready"], prompt_protocol=protocol)
        obj.update(ready=checked["ready"], preflight_receipt=entry, preflight_inventory=inventory)
        obj["qwen_binding_manifest"].update(special_ids_sha256=checked["special_ids_sha256"],
            vocabulary_size=checked["ready"]["vocabulary_size"], masked_tokens=checked["ready"]["masked_tokens"])
        dependencies.update((preflight_path, inventory["path"]))
        missing = []
    missing += ["final_dependency_freeze_and_exact_Git_readback", "separate_coordinator_enabled_activation",
                "unchanged_load_activation_gate", "owned_native_READY_resident_and_prompt_checks"]
    freeze = {"schema": "framed-interval-llm-v1-freeze-required-v1", "commit": None,
        "files_sha256": {name: sha_file(safe_path(root, name)) for name in sorted(dependencies)},
        "complete_preflight_dependencies": preflight_path is not None,
        "scope": "required current source/evidence bytes; not a Git readback receipt; resources are checked by frozen restorer"}
    mkdir_durable(destination, exclusive=True)
    atomic_json(destination / "activation-disabled.json", obj)
    atomic_json(destination / "freeze-required.json", freeze)
    status = {"schema": "framed-interval-llm-v1-run-activation-preparation-v1",
        "status": "disabled_preparation", "ready_for_inference": False,
        "planned_generations": 4, "planned_fresh_receivers": 2,
        "verified_resource_files": resources["verified_files"], "system_library_files": len(systems),
        "preflight_receipt_validated": preflight_path is not None, "remaining_gates": missing,
        "native_calls": 0, "tokenizer_calls": 0, "compile_calls": 0, "model_calls": 0,
        "activation": descriptor(root, output + "/activation-disabled.json"),
        "freeze_required": descriptor(root, output + "/freeze-required.json")}
    atomic_json(destination / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output, preflight_path=args.preflight_receipt,
        preflight_sha=args.preflight_sha256), sort_keys=True))


if __name__ == "__main__": main()
