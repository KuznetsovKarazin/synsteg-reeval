#!/usr/bin/env python3
"""Additive runtime provider for the frozen vocabulary-only implementation.

Requires separate SHA-pinned adapter Git readback for prepare/collect. This does
not bypass the original build, resource, source, plan, activation or resident
gates. No native process is started by fingerprint or prepare-activation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import (
    HELPER_PATH, current_runtime, load_vocab_activation, verify_adapter_readback,
)


def prepare_activation(args):
    # A fresh, private module instance receives exactly one explicit runtime
    # provider substitution. Frozen source bytes and every other check remain.
    spec = importlib.util.spec_from_file_location("synsteg_vocab_helper_runtime_compat_v1", ROOT / HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.current_runtime = current_runtime
    return helper.make_activation(ROOT, args.plan, args.plan_sha256, args.build, args.build_sha256,
        args.readback, args.readback_sha256, args.systems, args.systems_sha256, args.output,
        enable=args.enable_after_actual_build_and_git_readback, validator=load_vocab_activation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("fingerprint")
    prep = subs.add_parser("prepare-activation")
    collect = subs.add_parser("collect")
    for sub in (prep, collect):
        sub.add_argument("--adapter-readback", required=True)
        sub.add_argument("--adapter-readback-sha256", required=True)
        sub.add_argument("--output", required=True)
    for stem in ("plan", "build", "readback", "systems"):
        prep.add_argument("--"+stem, required=True)
        prep.add_argument("--"+stem+"-sha256", required=True)
    prep.add_argument("--enable-after-actual-build-and-git-readback", action="store_true")
    for name in ("activation", "activation-sha256", "intent"):
        collect.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        print(json.dumps(current_runtime(), sort_keys=True)); return 0
    verify_adapter_readback(ROOT, args.adapter_readback, args.adapter_readback_sha256)
    if args.command == "prepare-activation":
        print(json.dumps(prepare_activation(args), sort_keys=True)); return 0
    from synsteg_audit.framed_interval_llm_v1_vocab_collect import run_collection
    result = run_collection(ROOT, args.activation, args.activation_sha256, args.output,
        args.intent, activation_loader=load_vocab_activation)
    print(json.dumps({"state": result["state"], "planned_fixtures": 15, "planned_prompts": 2,
                      "forward_calls": 0, "runtime_identity_schema": "runtime-identity-compat-v1"}, sort_keys=True))
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "runtime_adapter_rejected", "type": type(exc).__name__,
                          "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
