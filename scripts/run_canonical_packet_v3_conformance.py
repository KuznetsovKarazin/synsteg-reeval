#!/usr/bin/env python3
"""Run the fixed v3 packet-tokenizer diagnostic after a verified Git freeze.

No model, torch, API, network, or generation calls. This runner has intentionally
not been executed while the protocol/code are still awaiting their Git freeze.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synsteg_audit.canonical_channel_v2 import ContractError, Prefix, PublicCodec
from synsteg_audit.canonical_packet_v3 import (
    CanonicalPacketGate, ConstructionRefusal, first_canonical_frontier, recover_packets,
)

PLAN = ROOT / "experiments/canonical-packet-v3/conformance-protocol.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def reference_path(gate, raw, provider, *, max_steps):
    """Sender-side path simulation knows its reference IDs, not payload bits."""
    ids = gate.codec.encode(raw)
    state, traces = Prefix(), []
    while len(state.ids) < len(ids):
        if len(traces) >= max_steps:
            return {"status": "reference_step_limit", "steps": traces}
        try:
            decision = gate.admit(state, provider(state.ids))
        except ConstructionRefusal as exc:
            return {"status": "construction_refusal", "reason": str(exc), "steps": traces}
        remaining = ids[len(state.ids):]
        matches = [row for row in decision.verdicts if row.weight > 0
                   and remaining[:len(row.packet)] == row.packet]
        trace = {"packet_index": len(traces), "token_offset": len(state.ids),
                 **decision.summary(), "reference_packet": None}
        traces.append(trace)
        if not matches:
            partial = any(row.weight > 0 and len(remaining) < len(row.packet)
                          and row.packet[:len(remaining)] == remaining for row in decision.verdicts)
            return {"status": "truncated_packet" if partial else "ineligible_packet", "steps": traces}
        if len(matches) != 1:
            raise AssertionError("Public raw frontier is not prefix-free")
        row = matches[0]
        trace["reference_packet"] = list(row.packet)
        if row.reason != "accepted":
            return {"status": "rejected_packet", "reason": row.reason, "steps": traces}
        state = gate.take(state, decision, row.packet)
    return {"status": "reference_path_accepted", "steps": traces,
            "reached_reference_bytes": state.raw == raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{40}", args.freeze_commit) is None:
        raise ValueError("Require the verified Git freeze commit")
    if args.output.exists():
        raise FileExistsError("Preserve any earlier v3 diagnostic result")
    protocol = json.loads(PLAN.read_text())
    if protocol["planned_fixture_count"] != 12 or len(protocol["fixtures"]) != 12:
        raise ValueError("All twelve predefined fixtures are required")
    limits = protocol["packet_policy"]
    if [limits[k] for k in ["max_packet_tokens", "node_budget", "max_reference_tokens_per_fixture",
                            "max_reference_packet_steps_per_fixture"]] != [5, 4096, 64, 64]:
        raise ValueError("Declared packet/budget/fixture limits changed")
    tokenizer_path = ROOT / protocol["tokenizer_relative_path"]
    raw_tokenizer = tokenizer_path.read_bytes()
    if sha(raw_tokenizer) != protocol["tokenizer_sha256"]:
        raise ValueError("Pinned tokenizer bytes changed")
    if importlib.metadata.version("tokenizers") != protocol["tokenizers_version"]:
        raise ValueError("Pinned tokenizer runtime changed")
    from tokenizers import Tokenizer
    specials = frozenset(row["id"] for row in json.loads(raw_tokenizer)["added_tokens"] if row["special"])
    source_paths = ["src/synsteg_audit/canonical_channel_v2.py", "src/synsteg_audit/canonical_packet_v3.py",
                    "tests/test_canonical_packet_v3.py", "scripts/run_canonical_packet_v3_conformance.py"]
    sources = {path: sha((ROOT / path).read_bytes()) for path in source_paths}
    resource = {"tokenizer_sha256": sha(raw_tokenizer), "tokenizers_version": protocol["tokenizers_version"],
                "adapter": protocol["adapter"], "special_ids": sorted(specials),
                "public_codec_source_sha256": sources[source_paths[0]]}
    resource_digest = sha(canonical(resource))

    def fresh_codec():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return PublicCodec(lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
            lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False).encode("utf-8", errors="strict"),
            specials, resource_digest)

    initial = fresh_codec()
    pool = {(token,) for token in set(range(259)) | set(specials)}
    for fixture in protocol["fixtures"]:
        ids = initial.encode(fixture["text"].encode("utf-8"))
        for start in range(len(ids)):
            for length in range(1, min(5, len(ids)-start) + 1):
                pool.add(ids[start:start+length])
    pool = tuple(sorted(pool))
    policy_digest = sha(canonical({"protocol_sha256": sha(PLAN.read_bytes()),
        "packet_source_sha256": sources[source_paths[1]], "public_path_pool": pool}))

    def fresh_gate():
        return CanonicalPacketGate(fresh_codec(), max_packet_tokens=5, policy_digest=policy_digest)

    def public_provider(gate, log):
        def provide(output_ids):
            boundary = Prefix(output_ids, gate.codec.render(output_ids))
            frontier = first_canonical_frontier(gate, boundary, pool, node_budget=4096)
            log.append({"prefix_ids": list(output_ids), **frontier.summary()})
            return frontier.candidates()
        return provide

    cases = []
    for fixture in protocol["fixtures"]:
        raw = fixture["text"].encode("utf-8")
        case = {"fixture_id": fixture["id"], "input_text": fixture["text"],
                "input_utf8_hex": raw.hex(), "input_sha256": sha(raw),
                "sender_constructions": [], "receiver_constructions": [], "compared_packet_steps": 0}
        try:
            sender = fresh_gate()
            ids = sender.codec.encode(raw)
            case["reference_token_ids"] = list(ids)
            case["tokenizer_roundtrip_exact"] = sender.codec.render(ids) == raw
            if len(ids) > 64:
                case.update({"sender": {"status": "reference_token_limit", "steps": []},
                    "receiver": {"status": "reference_token_limit", "steps": []},
                    "full_reference_path_accepted": False})
                cases.append(case)
                continue
            case["sender"] = reference_path(sender, raw,
                public_provider(sender, case["sender_constructions"]), max_steps=64)
            receiver = fresh_gate()
            case["receiver"] = recover_packets(receiver, raw,
                public_provider(receiver, case["receiver_constructions"]), expected_output_tokens=len(ids))
            if case["tokenizer_roundtrip_exact"]:
                if case["sender_constructions"] != case["receiver_constructions"]:
                    raise AssertionError("Sender and receiver public constructions differ")
                left, right = case["sender"]["steps"], case["receiver"]["steps"]
                if len(left) != len(right):
                    raise AssertionError("Canonical-wire receiver omitted a packet step")
                for a, b in zip(left, right):
                    aa = {k: v for k, v in a.items() if k != "reference_packet"}
                    bb = {k: v for k, v in b.items() if k != "observed_packet"}
                    if aa != bb or a["reference_packet"] != b["observed_packet"]:
                        raise AssertionError("Boundary eligibility or segmentation differs")
                    case["compared_packet_steps"] += 1
            case["full_reference_path_accepted"] = case["receiver"]["status"] == "canonical_packet_path_verified"
        except Exception as exc:
            # Preserve every predefined fixture even for an implementation/backend failure.
            # This is separately counted as an error, never recoded as canonical mass loss.
            case["error"] = {"type": type(exc).__name__, "message": str(exc)}
            case["full_reference_path_accepted"] = False
        cases.append(case)
    accepted = [case for case in cases if case["full_reference_path_accepted"]]
    errors = [case for case in cases if "error" in case]
    summary = {"schema": "canonical-packet-v3-conformance-result-v1", "experiment_id": "canonical-packet-v3",
        "freeze_commit": args.freeze_commit, "status": "completed_with_errors" if errors else "completed",
        "planned_fixtures": 12, "observed_fixtures": len(cases), "error_fixtures": len(errors),
        "verification_error_fixtures": sum(case["error"]["type"] == "AssertionError" for case in errors),
        "canonical_wire_inputs": sum(case.get("tokenizer_roundtrip_exact", False) for case in cases),
        "full_reference_paths_accepted_including_empty": len(accepted),
        "accepted_nonempty_paths": sum(bool(case["input_text"]) for case in accepted),
        "nonempty_fixtures": sum(bool(case["input_text"]) for case in cases),
        "refused_or_error_paths": len(cases)-len(accepted),
        "receiver_status_counts": dict(Counter(case.get("receiver", {}).get("status", "not_reached") for case in cases)),
        "compared_packet_steps": sum(case["compared_packet_steps"] for case in cases),
        "public_path_pool_size": len(pool), "max_packet_tokens": 5, "node_budget": 4096,
        "generated_texts": 0, "neural_forward_calls": 0, "paid_calls": 0, "closed_test_accessed": False,
        "input_weights_are_language_model_probabilities": False, "payload_delivery_or_security_validated": False,
        "limits": ["Known v2 fixtures reused for regression, not independent coverage evidence.",
                   "Artificial uniform law on all terminal frontier leaves; mass outside public trie unknown.",
                   "No model sampling, payload framing, authentication, or neural probability computation.",
                   "Fresh objects of the same pinned tokenizer implementation; not cross-backend reproducibility."]}
    inputs = {"protocol_sha256": sha(PLAN.read_bytes()), "freeze_commit": args.freeze_commit,
        "public_codec": resource, "codec_resource_digest": resource_digest,
        "policy_digest": policy_digest, "public_path_pool": [list(path) for path in pool],
        "public_path_pool_sha256": sha(canonical(pool)), "source_sha256": sources}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, obj in [("inputs.json", inputs), ("summary.json", summary), ("cases.json", cases)]:
        with (args.output / name).open("x") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    manifest = {path.name: sha(path.read_bytes()) for path in sorted(args.output.iterdir()) if path.is_file()}
    with (args.output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
