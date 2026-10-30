#!/usr/bin/env python3
"""runtime-identity-compat-v1 route for the frozen four-case launcher/workers.

Codec, journal, native service, reconciliation and fixed four-case protocol are
unchanged. Each fresh worker independently checks public adapter source hashes
and uses the explicit compatible runtime provider. Import does not launch.
"""
import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_activation import INTENT, checked_bytes, safe_path
from synsteg_audit.framed_interval_llm_v1_orchestration import run_campaign
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import (
    ENTRY_PATH, HELPER_PATH, MODULE_PATH, PINNED, load_run_activation, verify_adapter_readback,
)

SELF = "experiments/framed-interval-llm-v1/run_fixed512_runtime_compat_v1.py"
FROZEN_CLI = "experiments/framed-interval-llm-v1/run_fixed512.py"
FROZEN_ORCHESTRATION = "src/synsteg_audit/framed_interval_llm_v1_orchestration.py"
EXACT_PUBLIC_ADAPTER_PATHS = {SELF, MODULE_PATH, ENTRY_PATH, *PINNED, FROZEN_CLI, FROZEN_ORCHESTRATION}
ADDITIONAL_PINNED = {
    FROZEN_CLI: "30018c067b37d8f4f2e691b0d50bffeece752a326a5b3e218e4b9d044d1a6e93",
    FROZEN_ORCHESTRATION: "c17f0c1b537a562a4187192a8d7785489afdd17a7f17d97cbd9eb40c9ad0779c",
}


def verify_run_adapter(root, path, expected_sha):
    # Enforce the public-only allowlist BEFORE generic verification opens mapped
    # files. Receiver cannot be induced to hash private proposal/sender files.
    obj = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if type(obj.get("files_sha256")) is not dict or set(obj["files_sha256"]) != EXACT_PUBLIC_ADAPTER_PATHS:
        raise ValueError("exact public seven-file run-adapter readback required")
    proof = verify_adapter_readback(root, path, expected_sha)
    if any(proof["files_sha256"][name] != expected for name, expected in ADDITIONAL_PINNED.items()):
        raise ValueError("frozen worker/orchestration source differs")
    return proof


def spawn_worker(root, activation_path, activation_sha, campaign_relative, endpoint_relative,
                 request_relative, timeout, *, adapter_readback, adapter_readback_sha256):
    # Lifecycle matches frozen orchestration.spawn_worker; only entrypoint and
    # two source-readback arguments differ. Native child remains in owned group.
    started = time.monotonic()
    root = Path(root)
    endpoint = safe_path(root, endpoint_relative)
    command = [sys.executable, str(root / SELF), "worker",
        "--activation", activation_path, "--activation-sha256", activation_sha, "--intent", INTENT,
        "--campaign", campaign_relative, "--endpoint", endpoint_relative, "--request", request_relative,
        "--adapter-readback", adapter_readback, "--adapter-readback-sha256", adapter_readback_sha256]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    with (endpoint / "worker.stdout.log").open("xb") as stdout, (endpoint / "worker.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            outcome = {"status": "exited", "exit_code": code, "pid": process.pid}
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
            outcome = {"status": "timeout", "exit_code_before_cleanup": process.returncode, "pid": process.pid}
        finally:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=3)
        outcome["exit_code"] = process.returncode
        outcome["wall_seconds"] = time.monotonic()-started
        return outcome


def worker(args):
    spec = importlib.util.spec_from_file_location("synsteg_fixed512_worker_runtime_compat_v1", ROOT / FROZEN_CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_activation = load_run_activation
    return module.worker(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        sub = subs.add_parser(name)
        for argument in ("activation", "activation-sha256", "intent", "adapter-readback", "adapter-readback-sha256"):
            sub.add_argument("--"+argument, required=True)
        if name == "run": sub.add_argument("--output", required=True)
        else:
            for argument in ("campaign", "endpoint", "request"): sub.add_argument("--"+argument, required=True)
    args = parser.parse_args(argv)
    verify_run_adapter(ROOT, args.adapter_readback, args.adapter_readback_sha256)
    if args.command == "worker": return worker(args)
    runner = functools.partial(spawn_worker, adapter_readback=args.adapter_readback,
                               adapter_readback_sha256=args.adapter_readback_sha256)
    ledger = run_campaign(ROOT, args.activation, args.activation_sha256, args.output, args.intent,
        activation_loader=load_run_activation, worker_runner=runner)
    print(json.dumps({"state": ledger["state"], "planned_cases": 4,
        "authenticated_payload_deliveries": ledger["authenticated_payload_deliveries"],
        "authenticated_delivery_denominator": 2, "runtime_identity_schema": "runtime-identity-compat-v1"}, sort_keys=True))
    return 0 if ledger["state"] == "finished_all_planned_outcomes" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "runtime_adapter_launcher_rejected", "type": type(exc).__name__,
                          "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
