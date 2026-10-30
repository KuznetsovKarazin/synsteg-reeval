#!/usr/bin/env python3
"""One new fixed vocabulary attempt using the validated pidfd ownership adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_activation import checked_bytes
from synsteg_audit.framed_interval_llm_v1_vocab_collect import INTENT as BASE_INTENT, run_collection
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import load_vocab_activation
from synsteg_audit.proc_namespace_identity_v2 import capture_owned_child

SELF = "experiments/framed-interval-llm-v1/vocab_proc_namespace_identity_v2.py"
MODULE = "src/synsteg_audit/proc_namespace_identity_v2.py"
PRIOR_ENTRY = "experiments/framed-interval-llm-v1/vocab_proc_namespace_identity_v1.py"
PLAN = "experiments/framed-interval-llm-v1/proc-namespace-identity-v2-plan.json"
PINNED = {
    PLAN: "3e9dcca4f263c49fe5540642dd473dc98a1776598fdc9cb1cf7c0fd2f276b26e",
    PRIOR_ENTRY: "ce66eb70f18b35130446c9cd21a2aec2f27f3643e6d4d3cef9f000f38077bef0",
    "src/synsteg_audit/proc_namespace_identity_v1.py": "30ced033d3f5186e41f1b99720f2057287ec2ed6172483fc46750ad797aaea94",
    "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py": "6ef38d3b3d88da78e3ed587f0582aa5f31d11fa26973260c841ef8843c4fd938",
    "src/synsteg_audit/framed_interval_llm_v1_vocab_collect.py": "0ee0aa5b4631cb6750f7a9084cba81234626d9de42be9d3a14efe66814863ab4",
}
INTEGRATION_FILES = {"results/proc-namespace-identity-v2-integration/" + name
                     for name in ("plan.json", "result.json", "reproduce.py")}
EXACT_PATHS = {SELF, MODULE, *PINNED, *INTEGRATION_FILES}


def load_readback(root, path, expected_sha):
    proof = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if (proof.get("schema") != "proc-namespace-identity-v2-git-readback"
            or type(proof.get("commit")) is not str
            or re.fullmatch("[0-9a-f]{40}", proof["commit"]) is None
            or proof.get("all_files_exact") is not True
            or type(proof.get("files_sha256")) is not dict
            or set(proof["files_sha256"]) != EXACT_PATHS):
        raise ValueError("exact ten-file pidfd adapter/integration Git readback required")
    for name, expected in proof["files_sha256"].items():
        checked_bytes(root, {"path": name, "sha256": expected})
    if any(proof["files_sha256"][name] != expected for name, expected in PINNED.items()):
        raise ValueError("fixed pidfd dependency differs")
    plan = json.loads(checked_bytes(root, {"path": PLAN, "sha256": PINNED[PLAN]}))
    for entry in plan["prior_failed_attempts"]:
        if json.loads(checked_bytes(root, entry)).get("state") != "failed_no_retry":
            raise ValueError("prior failed attempts must remain preserved")
    result = json.loads(checked_bytes(root, plan["harmless_integration"]))
    integration_plan = json.loads(checked_bytes(root, plan["harmless_integration_plan"]))
    if (result.get("status") != "passed" or result.get("model_calls") != 0
            or result.get("tokenizer_calls") != 0 or result.get("child_returncode") != 0
            or result.get("cleanup_errors") != [] or result.get("within_budget") is not True
            or result.get("identity", {}).get("schema") != "proc-namespace-identity-v2"
            or integration_plan["files_sha256"][MODULE] != proof["files_sha256"][MODULE]):
        raise ValueError("actual harmless integration for this resolver is required")
    for name, expected in integration_plan["files_sha256"].items():
        checked_bytes(root, {"path": name, "sha256": expected})
    return proof, plan


def owned_bindings(root, binary_sha, readback_identity):
    checked_bytes(root, {"path": PRIOR_ENTRY, "sha256": PINNED[PRIOR_ENTRY]})
    spec = importlib.util.spec_from_file_location("synsteg_vocab_owned_transport_pidfd_v2", root / PRIOR_ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.capture_owned_child = capture_owned_child
    return module.owned_bindings(root, binary_sha, readback_identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("readback", "readback-sha256", "intent"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    _, plan = load_readback(ROOT, args.readback, args.readback_sha256)
    if args.intent != plan["intent"]: raise ValueError("explicit new pidfd vocabulary attempt intent required")
    entry = plan["activation"]
    activation = json.loads(checked_bytes(ROOT, entry))
    factory, resident = owned_bindings(ROOT, activation["binary"]["sha256"],
        {"path": args.readback, "sha256": args.readback_sha256})
    result = run_collection(ROOT, entry["path"], entry["sha256"], plan["output"], BASE_INTENT,
        activation_loader=load_vocab_activation, transport_factory=factory, resident_verifier=resident)
    print(json.dumps({"state": result["state"], "experiment_id": plan["experiment_id"],
        "planned_fixtures": 15, "planned_prompts": 2, "forward_calls": 0,
        "prior_failed_attempts_preserved": 2}, sort_keys=True))
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "pidfd_adapter_rejected", "type": type(exc).__name__,
                          "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
