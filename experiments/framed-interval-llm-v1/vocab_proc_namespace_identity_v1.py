#!/usr/bin/env python3
"""One separately frozen vocabulary preflight with owned procfs PID binding.

This is a new engineering attempt; native-v1 remains failed and is never
overwritten. All fifteen fixtures/two prompts and the native120s budget remain.
The old collector and compatible runtime loader are used without source edits.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.framed_interval_llm_v1_activation import checked_bytes, safe_path
from synsteg_audit.framed_interval_llm_v1_journal import atomic_json
from synsteg_audit.framed_interval_llm_v1_vocab_collect import (
    INTENT as BASE_INTENT, NativeTransport, run_collection, verify_collection_resident,
)
from synsteg_audit.framed_interval_llm_runtime_compat_v1 import load_vocab_activation
from synsteg_audit.proc_namespace_identity_v1 import capture_owned_child

SELF = "experiments/framed-interval-llm-v1/vocab_proc_namespace_identity_v1.py"
MODULE = "src/synsteg_audit/proc_namespace_identity_v1.py"
PLAN = "experiments/framed-interval-llm-v1/proc-namespace-identity-v1-plan.json"
PLAN_SHA = "51c0894b45c3e9020c68aa9abb846e2f654ce14d91e76aa63803053167364986"
PINNED = {
    "src/synsteg_audit/framed_interval_llm_runtime_compat_v1.py": "6ef38d3b3d88da78e3ed587f0582aa5f31d11fa26973260c841ef8843c4fd938",
    "experiments/framed-interval-llm-v1/vocab_runtime_compat_v1.py": "03b2ca3959844f4c4a51e873f6b698e7139e21a4296cd09ceaa0fcf1adbdd650",
    "scripts/prepare_framed_interval_llm_vocab_activation.py": "b4d92d91acaf29b8a7d150739cad232d3b0e5c7f8d3cc0d540de382951706fdb",
    "src/synsteg_audit/framed_interval_llm_v1_vocab_collect.py": "0ee0aa5b4631cb6750f7a9084cba81234626d9de42be9d3a14efe66814863ab4",
    PLAN: PLAN_SHA,
}


def load_readback(root, path, expected_sha):
    import re
    proof = json.loads(checked_bytes(root, {"path": path, "sha256": expected_sha}))
    if (proof.get("schema") != "proc-namespace-identity-v1-git-readback"
            or type(proof.get("commit")) is not str
            or re.fullmatch("[0-9a-f]{40}", proof["commit"]) is None
            or proof.get("all_files_exact") is not True
            or type(proof.get("files_sha256")) is not dict
            or set(proof["files_sha256"]) != {SELF, MODULE, *PINNED}):
        raise ValueError("exact separately frozen seven-file proc adapter readback required")
    for name, expected in proof["files_sha256"].items():
        checked_bytes(root, {"path": name, "sha256": expected})
    if any(proof["files_sha256"][name] != expected for name, expected in PINNED.items()):
        raise ValueError("fixed proc adapter dependency differs")
    plan = json.loads(checked_bytes(root, {"path": PLAN, "sha256": PLAN_SHA}))
    checked_bytes(root, {k: plan["baseline_failed_attempt"][k] for k in ("path", "sha256")})
    return proof, plan


class OwnedTransport:
    def __init__(self, base, ticket):
        self.base, self.ticket, self.process = base, ticket, base.process

    def __getattr__(self, name): return getattr(self.base, name)

    def close(self):
        try: self.base.close()
        finally: self.ticket.close()


def owned_bindings(root, binary_sha, readback_identity):
    """Fresh closure per run, with at most one actual transport construction."""
    holder = {}

    def factory(binary, model, libraries, directory, deadline):
        if holder.get("attempted"): raise ValueError("owned transport already attempted; no retry")
        holder["attempted"] = True
        base = NativeTransport(binary, model, libraries, directory, deadline)
        ticket = None
        try:
            ticket = capture_owned_child(base.process, binary, binary_sha)
            capture = {"schema": "proc-namespace-identity-v1-capture", "popen_pid": base.process.pid,
                "parent": ticket.parent, "child": ticket.child,
                "executable": ticket.expected_executable, "adapter_readback": readback_identity}
            atomic_json(Path(directory) / "proc-identity-capture.json", capture)
            wrapped = OwnedTransport(base, ticket)
            holder["transport"] = wrapped
            return wrapped
        except BaseException:
            try: base.close()
            finally:
                if ticket is not None: ticket.close()
            raise

    def resident(root_arg, activation, report, pid):
        transport = holder.get("transport")
        if transport is None or transport.process.pid != pid:
            raise ValueError("resident verification has no matching owned Popen ticket")
        maps, identity = transport.ticket.maps_snapshot()
        result = verify_collection_resident(root_arg, activation, report, pid, maps_text=maps)
        final = transport.ticket.snapshot()
        if final != identity["after"]: raise ValueError("child changed during resident verification")
        return {**result, "proc_namespace_identity": identity,
            "post_resident_identity": final, "adapter_readback": readback_identity}

    return factory, resident


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("readback", "readback-sha256", "intent"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    _, plan = load_readback(ROOT, args.readback, args.readback_sha256)
    if args.intent != plan["intent"]: raise ValueError("explicit new proc-namespace attempt intent required")
    activation_entry = plan["activation"]
    activation = json.loads(checked_bytes(ROOT, activation_entry))
    identity = {"path": args.readback, "sha256": args.readback_sha256}
    factory, resident = owned_bindings(ROOT, activation["binary"]["sha256"], identity)
    result = run_collection(ROOT, activation_entry["path"], activation_entry["sha256"],
        plan["output"], BASE_INTENT, activation_loader=load_vocab_activation,
        transport_factory=factory, resident_verifier=resident)
    print(json.dumps({"state": result["state"], "experiment_id": plan["experiment_id"],
        "planned_fixtures": 15, "planned_prompts": 2, "forward_calls": 0,
        "prior_failed_attempt_preserved": True}, sort_keys=True))
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "proc_namespace_adapter_rejected", "type": type(exc).__name__,
                          "message": str(exc)[:1024]}), file=sys.stderr)
        raise SystemExit(2)
