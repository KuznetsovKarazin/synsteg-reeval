#!/usr/bin/env python3
"""Materialize build evidence/activation after coordinator actions; never compile or launch.

An enabled activation requires an existing exact binary, a successful build
receipt, a SHA-pinned coordinator Git readback and an explicit flag. This script
does not produce its own Git approval and never calls the collector process.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_activation import checked_bytes, current_runtime, safe_path, sha_file
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json, fsync_directory
from synsteg_audit.framed_interval_llm_v1_preflight import canonical_json
from synsteg_audit.framed_interval_llm_v1_vocab_collect import (
    PLAN_SHA256, PUBLIC_PATH, PUBLIC_SHA, RUNTIME_COMMIT, SOURCE_PATHS, digest,
    load_collection_activation,
)

SCHEMA = "framed-interval-llm-v1-vocab-concrete-build-plan-v1"
SCRIPT_PATH = "scripts/prepare_framed_interval_llm_vocab_activation.py"


def descriptor(root, name):
    return {"path": name, "sha256": sha_file(safe_path(root, name))}


def external_identity(entry):
    if (type(entry) is not dict or set(entry) != {"path", "sha256"}
            or type(entry["path"]) is not str or not Path(entry["path"]).is_absolute()
            or not re.fullmatch("[0-9a-f]{64}", entry.get("sha256", ""))
            or sha_file(entry["path"]) != entry["sha256"]):
        raise ValueError("current external file identity differs")


def load_plan(root, path, expected_sha):
    plan = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if (plan.get("schema") != SCHEMA or plan.get("state") != "planned_not_compiled"
            or plan.get("runtime_commit") != RUNTIME_COMMIT
            or plan.get("resource_root") != "resources/seeded-pair-llm-v1"
            or set(plan.get("sources_sha256", {})) != set(SOURCE_PATHS)):
        raise ValueError("exact concrete source/resource plan required")
    if plan["source_bundle_sha256"] != digest(canonical_json(plan["sources_sha256"])):
        raise ValueError("collector source bundle digest differs")
    for name, expected in plan["sources_sha256"].items(): checked_bytes(root, {"path": name, "sha256": expected})
    for name, expected in plan["headers"].items(): checked_bytes(root, {"path": name, "sha256": expected})
    if len(plan["headers"]) != 7: raise ValueError("complete7-header closure required")
    checked_bytes(root, plan["header_provenance"])
    checked_bytes(root, plan["upstream_license"])
    checked_bytes(root, plan["static_dependency_review"])
    checked_bytes(root, plan["system_library_manifest"])
    checked_bytes(root, plan["preparation_script"])
    if plan["preparation_script"]["path"] != SCRIPT_PATH: raise ValueError("wrong activation preparer")
    for entry in [plan["compiler"], *plan["compiler_subtools"].values()]: external_identity(entry)
    command = plan["build_command"]
    if (type(command) is not list or not command or any(type(arg) is not str or not arg for arg in command)
            or command[0] != plan["compiler"]["path"] or command[-2:] != ["-o", plan["binary_path"]]
            or plan["build_environment"] != {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
            or plan["working_directory"] != "repository_root" or plan["compile_timeout_seconds"] != 120
            or plan["compile_attempts"] != 1):
        raise ValueError("concrete compiler command/budget differs")
    for name, expected in plan["link_resource_files_sha256"].items(): checked_bytes(root, {"path": name, "sha256": expected})
    resource = json.loads(checked_bytes(root, plan["resource_restore_receipt"]))
    if (resource.get("status") != "verified" or resource.get("verified_files") != 41
            or resource.get("frozen_resource_identity_sha256") != plan["resource_identity_sha256"]):
        raise ValueError("old41 resource restoration receipt differs")
    if plan["tokenizer_plan"] != {"path": "experiments/framed-interval-llm-v1/tokenizer-preflight-proposal.json", "sha256": PLAN_SHA256}:
        raise ValueError("fixed tokenizer preflight plan differs")
    checked_bytes(root, plan["tokenizer_plan"])
    return plan


def binary_identity(root, plan):
    path = safe_path(root, plan["binary_path"])
    if not path.is_file() or not os.access(path, os.X_OK): raise ValueError("actual executable collector binary required")
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF": raise ValueError("actual ELF collector binary required")
    return descriptor(root, plan["binary_path"])


def validate_execution(root, plan, plan_sha, evidence):
    execution = json.loads(checked_bytes(root, evidence))
    if (execution.get("schema") != "framed-interval-llm-v1-vocab-build-execution-v1"
            or execution.get("plan_sha256") != plan_sha or execution.get("build_command") != plan["build_command"]
            or execution.get("build_environment") != plan["build_environment"]
            or execution.get("working_directory") != "repository_root"
            or type(execution.get("exit_code")) is not int or execution["exit_code"] != 0
            or type(execution.get("compile_attempts")) is not int or execution["compile_attempts"] != 1
            or type(execution.get("native_model_executions")) is not int or execution["native_model_executions"] != 0
            or type(execution.get("wall_seconds")) not in {int, float}
            or not 0 <= execution["wall_seconds"] <= plan["compile_timeout_seconds"]
            or not re.fullmatch("[0-9a-f]{40}", execution.get("source_freeze_commit", ""))):
        raise ValueError("successful one-attempt build evidence required")
    for name in ("stdout", "stderr"): checked_bytes(root, execution[name])
    binary = binary_identity(root, plan)
    if execution.get("binary") != binary: raise ValueError("actual binary differs from compiler evidence")
    return execution, binary


def make_build_receipt(root, plan_path, plan_sha, execution_path, execution_sha, output_path):
    plan = load_plan(root, plan_path, plan_sha)
    execution, binary = validate_execution(root, plan, plan_sha, {"path": execution_path, "sha256": execution_sha})
    receipt = {"schema": "framed-interval-llm-v1-vocab-build-v1",
        "source_bundle_sha256": plan["source_bundle_sha256"], "runtime_commit": RUNTIME_COMMIT,
        "binary_sha256": binary["sha256"], "native_model_executions": 0,
        "compiler": plan["compiler"], "headers": plan["headers"], "build_command": plan["build_command"],
        "build_plan": {"path": plan_path, "sha256": plan_sha},
        "execution_evidence": {"path": execution_path, "sha256": execution_sha},
        "compiler_subtools": plan["compiler_subtools"], "header_provenance": plan["header_provenance"],
        "scope": "recorded coordinator compilation evidence and current files; no binary/model execution"}
    target = safe_path(root, output_path)
    if not target.parent.is_dir(): raise ValueError("create receipt directory before publication")
    atomic_json(target, receipt)
    return descriptor(root, output_path)


def make_activation(root, plan_path, plan_sha, build_path, build_sha, readback_path, readback_sha,
                    systems_path, systems_sha, output_path, *, enable=False, validator=load_collection_activation):
    if type(enable) is not bool: raise ValueError("explicit boolean activation flag required")
    plan = load_plan(root, plan_path, plan_sha)
    build = json.loads(checked_bytes(root, {"path": build_path, "sha256": build_sha}))
    binary = binary_identity(root, plan)
    if (build.get("schema") != "framed-interval-llm-v1-vocab-build-v1"
            or build.get("source_bundle_sha256") != plan["source_bundle_sha256"]
            or build.get("binary_sha256") != binary["sha256"] or build.get("build_command") != plan["build_command"]
            or build.get("headers") != plan["headers"] or build.get("compiler") != plan["compiler"]
            or build.get("runtime_commit") != RUNTIME_COMMIT or type(build.get("native_model_executions")) is not int
            or build["native_model_executions"] != 0
            or build.get("build_plan") != {"path": plan_path, "sha256": plan_sha}
            or build.get("compiler_subtools") != plan["compiler_subtools"]
            or build.get("header_provenance") != plan["header_provenance"]):
        raise ValueError("actual build receipt differs from frozen plan")
    execution, _ = validate_execution(root, plan, plan_sha, build["execution_evidence"])
    proof = json.loads(checked_bytes(root, {"path": readback_path, "sha256": readback_sha}))
    if (proof.get("schema") != "framed-interval-llm-v1-vocab-build-git-readback-v1"
            or type(proof.get("commit")) is not str or not re.fullmatch("[0-9a-f]{40}", proof["commit"])
            or proof.get("all_files_exact") is not True or type(proof.get("files_sha256")) is not dict):
        raise ValueError("coordinator actual Git readback required")
    required = {**plan["sources_sha256"], **plan["headers"], plan_path: plan_sha, build_path: build_sha,
        binary["path"]: binary["sha256"], plan["header_provenance"]["path"]: plan["header_provenance"]["sha256"],
        plan["upstream_license"]["path"]: plan["upstream_license"]["sha256"],
        SCRIPT_PATH: plan["preparation_script"]["sha256"], plan["tokenizer_plan"]["path"]: plan["tokenizer_plan"]["sha256"],
        systems_path: systems_sha,
        build["execution_evidence"]["path"]: build["execution_evidence"]["sha256"],
        execution["stdout"]["path"]: execution["stdout"]["sha256"],
        execution["stderr"]["path"]: execution["stderr"]["sha256"],
        PUBLIC_PATH: PUBLIC_SHA}
    if any(proof["files_sha256"].get(name) != value for name, value in required.items()):
        raise ValueError("Git readback omits or changes plan/source/header/binary/build artifacts")
    for name, expected in required.items(): checked_bytes(root, {"path": name, "sha256": expected})
    if plan["system_library_manifest"] != {"path": systems_path, "sha256": systems_sha}:
        raise ValueError("system-library declaration differs from reviewed build plan")
    systems = json.loads(checked_bytes(root, {"path": systems_path, "sha256": systems_sha}))
    if systems.get("schema") != "framed-interval-llm-v1-vocab-system-libraries-v1":
        raise ValueError("static system-library manifest required")
    mapping = systems.get("files_sha256")
    if type(mapping) is not dict or not mapping: raise ValueError("nonempty system-library map required")
    for name, expected in mapping.items(): external_identity({"path": name, "sha256": expected})
    activation = {"schema": "framed-interval-llm-v1-vocab-activation-v1", "enabled": bool(enable),
        "freeze_commit": proof["commit"], "coordinator_readback": {"commit": proof["commit"], "all_files_exact": True},
        "sources_sha256": plan["sources_sha256"], "plan": plan["tokenizer_plan"],
        "public_protocol": {"path": PUBLIC_PATH, "sha256": PUBLIC_SHA}, "binary": binary,
        "build_receipt": {"path": build_path, "sha256": build_sha}, "resource_identity_sha256": plan["resource_identity_sha256"],
        "runtime": current_runtime(), "native_system_libraries_sha256": mapping, "timeout_seconds": 120}
    target = safe_path(root, output_path)
    if not target.parent.is_dir(): raise ValueError("create activation directory before publication")
    if enable:
        # This is only the frozen read-only validator, never run_collection/Popen.
        candidate_path = output_path+".validation-candidate.json"
        candidate = safe_path(root, candidate_path)
        atomic_json(candidate, activation)
        candidate_sha = sha_file(candidate)
        validator(root, candidate_path, candidate_sha, "collect-frozen-qwen-vocabulary-only-preflight")
        os.link(candidate, target)  # Publish exclusively only after successful validation.
        candidate.unlink(); fsync_directory(target.parent)
    else:
        atomic_json(target, activation)
    return descriptor(root, output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("receipt", "activation"):
        p = sub.add_parser(name)
        p.add_argument("--plan", required=True); p.add_argument("--plan-sha256", required=True)
        p.add_argument("--output", required=True)
        if name == "receipt":
            p.add_argument("--execution", required=True); p.add_argument("--execution-sha256", required=True)
        else:
            for stem in ("build", "readback", "systems"):
                p.add_argument("--"+stem, required=True); p.add_argument("--"+stem+"-sha256", required=True)
            p.add_argument("--enable-after-actual-build-and-git-readback", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "receipt":
        result = make_build_receipt(ROOT, args.plan, args.plan_sha256, args.execution, args.execution_sha256, args.output)
    else:
        result = make_activation(ROOT, args.plan, args.plan_sha256, args.build, args.build_sha256,
            args.readback, args.readback_sha256, args.systems, args.systems_sha256, args.output,
            enable=args.enable_after_actual_build_and_git_readback)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
