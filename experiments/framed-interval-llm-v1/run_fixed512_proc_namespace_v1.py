#!/usr/bin/env python3
"""Explicit fixed-four-case route with runtime and owned procfs compatibility.

Import has no native side effects. Every parent/fresh worker verifies the exact
public adapter dependency map; frozen activation/resource/preflight gates still
apply. No private proposal or sender traces appear in this readback manifest.
"""
import argparse
import functools
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_activation import checked_bytes
from synsteg_audit.framed_interval_llm_v1_orchestration import execute_endpoint, run_campaign
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import load_run_activation
from synsteg_audit.framed_interval_llm_proc_service_v1 import ProcNativeService

SELF = "experiments/framed-interval-llm-v1/run_fixed512_proc_namespace_v1.py"
MODULE = "src/synsteg_audit/framed_interval_llm_proc_service_v1.py"
FROZEN_CLI = "experiments/framed-interval-llm-v1/run_fixed512.py"
ROUTER = "experiments/framed-interval-llm-v1/run_fixed512_runtime_compat_v1.py"
PINNED = {
    FROZEN_CLI: "30018c067b37d8f4f2e691b0d50bffeece752a326a5b3e218e4b9d044d1a6e93",
    ROUTER: "cebc2ee7f6b6e2ee0aacfa9fcd9cb71f417f3374bdf6dafe544dd8b837851f18",
    "src/synsteg_audit/seeded_pair_llm.py": "687e5c74b7b3f6675e2d057d13146f3ffa716a58fbf77522830b58b023ba89f0",
    "src/synsteg_audit/framed_interval_llm_v1_orchestration.py": "c17f0c1b537a562a4187192a8d7785489afdd17a7f17d97cbd9eb40c9ad0779c",
    "src/synsteg_audit/framed_interval_llm_v1_activation.py": "262f9aff1f12e421031ef5e9c961229df986bf6bb4c69f96e0557129d830340d",
    "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py": "6ef38d3b3d88da78e3ed587f0582aa5f31d11fa26973260c841ef8843c4fd938",
    "src/synsteg_audit/proc_namespace_identity_v1.py": "30ced033d3f5186e41f1b99720f2057287ec2ed6172483fc46750ad797aaea94",
    "src/synsteg_audit/proc_namespace_identity_v2.py": "ee5714058205b295c50fe8cb3cce499d1a3b6fa4527f333fd1205c5cf5c52a6d",
}
EXACT_PUBLIC_PATHS = {SELF, MODULE, *PINNED}


def verify_readback(root, path, expected_sha):
    proof = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if (proof.get("schema") != "framed-interval-llm-proc-service-v1-git-readback"
            or type(proof.get("commit")) is not str
            or re.fullmatch("[0-9a-f]{40}", proof["commit"]) is None
            or proof.get("all_files_exact") is not True
            or type(proof.get("files_sha256")) is not dict
            or set(proof["files_sha256"]) != EXACT_PUBLIC_PATHS):
        raise ValueError("exact public ten-file proc-service adapter readback required")
    # Check the public-only shape before opening any path from this mapping.
    for name, expected in proof["files_sha256"].items():
        checked_bytes(root, {"path": name, "sha256": expected})
    if any(proof["files_sha256"][name] != expected for name, expected in PINNED.items()):
        raise ValueError("frozen proc-service dependency identity differs")
    return proof


def _private_module(path, name):
    checked_bytes(ROOT, {"path": path, "sha256": PINNED[path]})
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    module = _private_module(FROZEN_CLI, "synsteg_fixed512_proc_service_worker_v1")
    module.load_activation = load_run_activation
    factory = functools.partial(ProcNativeService, adapter_readback={
        "path": args.adapter_readback, "sha256": args.adapter_readback_sha256})
    module.execute_endpoint = functools.partial(execute_endpoint, service_factory=factory)
    return module.worker(args)


def worker_runner(args):
    module = _private_module(ROUTER, "synsteg_fixed512_proc_service_router_v1")
    module.SELF = SELF  # Only the new fresh-worker entrypoint changes.
    return functools.partial(module.spawn_worker, adapter_readback=args.adapter_readback,
                             adapter_readback_sha256=args.adapter_readback_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        sub = subs.add_parser(name)
        for arg in ("activation", "activation-sha256", "intent", "adapter-readback", "adapter-readback-sha256"):
            sub.add_argument("--" + arg, required=True)
        if name == "run": sub.add_argument("--output", required=True)
        else:
            for arg in ("campaign", "endpoint", "request"): sub.add_argument("--" + arg, required=True)
    args = parser.parse_args(argv)
    verify_readback(ROOT, args.adapter_readback, args.adapter_readback_sha256)
    if args.command == "worker": return worker(args)
    ledger = run_campaign(ROOT, args.activation, args.activation_sha256, args.output, args.intent,
        activation_loader=load_run_activation, worker_runner=worker_runner(args))
    print(json.dumps({"state": ledger["state"], "planned_cases": 4,
        "authenticated_payload_deliveries": ledger["authenticated_payload_deliveries"],
        "authenticated_delivery_denominator": 2, "proc_adapter": "framed-interval-llm-proc-service-v1"}, sort_keys=True))
    return 0 if ledger["state"] == "finished_all_planned_outcomes" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "proc_service_launcher_rejected", "type": type(exc).__name__,
                          "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
