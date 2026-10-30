#!/usr/bin/env python3
"""Explicit one-attempt bounded16+8 route; refuses absent/mismatched Git activation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from synsteg_audit.framed_interval_bounded_v1_activation import load_activation
from synsteg_audit.framed_interval_llm_v1_activation import safe_path
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json
from synsteg_audit.framed_interval_bounded_v1_orchestration import execute_endpoint, run_campaign


def worker(args):
    # Check location without opening any resources or native process.
    campaign = safe_path(ROOT, args.campaign)
    directory = safe_path(ROOT, args.endpoint)
    request_path = safe_path(ROOT, args.request)
    request = json.loads(request_path.read_bytes())
    if not isinstance(request, dict) or request.get("role") not in {"generate", "receive"}:
        raise ValueError("valid endpoint request required")
    role_dir = "sender" if request["role"] == "generate" else "receiver"
    if (not args.campaign.startswith("results/framed-interval-bounded-v1/")
            or directory != campaign / request.get("run_id", "") / role_dir
            or request_path != directory / "request.json" or not directory.is_dir()):
        raise ValueError("endpoint/request outside exact campaign layout")
    if (directory / "execution-started.json").exists() or (directory / "worker-result.json").exists():
        raise ValueError("one-attempt endpoint already exists; retry forbidden")
    try:
        activation = load_activation(ROOT, args.activation, args.activation_sha256, args.intent,
                                     receiver_scope=request["role"] == "receive")
        result = execute_endpoint(request, activation, campaign, directory)
        return 0 if result["status"] == "returned" else 2
    except Exception as exc:
        # A prelaunch or unexpected Python failure is a process outcome, never a
        # negative cryptographic result. Existing terminal records are preserved.
        target = directory / "worker-result.json"
        if not target.exists():
            atomic_json(target, {"schema": "framed-interval-bounded-v1-worker-result-v1",
                "run_id": request.get("run_id"), "role": request["role"],
                "execution_kind": "native", "activation_sha256": args.activation_sha256,
                "status": "prelaunch_error" if not (directory / "execution-started.json").exists() else "worker_exception",
                "retry_permitted": False, "core": None,
                "error": {"type": type(exc).__name__, "message": str(exc)[:1024]}})
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "worker"):
        sub = subs.add_parser(command)
        sub.add_argument("--activation", required=True)
        sub.add_argument("--activation-sha256", required=True)
        sub.add_argument("--intent", required=True)
        if command == "run": sub.add_argument("--output", required=True)
        else:
            sub.add_argument("--campaign", required=True)
            sub.add_argument("--endpoint", required=True)
            sub.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    if args.command == "worker": return worker(args)
    ledger = run_campaign(ROOT, args.activation, args.activation_sha256, args.output, args.intent)
    print(json.dumps({"state": ledger["state"], "planned_cases": 16,
                      "authenticated_payload_deliveries": ledger["authenticated_payload_deliveries"],
                      "authenticated_delivery_denominator": 8}, sort_keys=True))
    return 0 if ledger["state"] == "finished_all_planned_outcomes" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "launcher_rejected", "type": type(exc).__name__, "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
