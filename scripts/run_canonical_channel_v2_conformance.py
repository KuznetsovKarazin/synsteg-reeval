#!/usr/bin/env python3
"""Run the fixed tokenizer-only diagnostic; no torch, model or network imports."""
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
from synsteg_audit.canonical_channel_v2 import CanonicalGate, Prefix, PublicCodec, recover_path

PLAN = ROOT / "experiments/canonical-channel-v2/conformance-protocol.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    if re.fullmatch("[0-9a-f]{40}", args.freeze_commit) is None:
        raise ValueError("Require the verified Git freeze commit")
    if args.output.exists():
        raise FileExistsError("Preserve the earlier conformance result")
    protocol = json.loads(PLAN.read_text())
    if protocol["planned_fixture_count"] != len(protocol["fixtures"]) or len(protocol["fixtures"]) != 12:
        raise ValueError("The full fixed twelve-fixture denominator is required")
    tokenizer_path = ROOT / protocol["tokenizer_relative_path"]
    raw_tokenizer = tokenizer_path.read_bytes()
    if sha(raw_tokenizer) != protocol["tokenizer_sha256"]:
        raise ValueError("Tokenizer resource identity changed")
    if importlib.metadata.version("tokenizers") != protocol["tokenizers_version"]:
        raise ValueError("Tokenizer runtime changed")
    from tokenizers import Tokenizer
    special_ids = frozenset(row["id"] for row in json.loads(raw_tokenizer)["added_tokens"] if row["special"])
    resource = {"tokenizer_sha256": sha(raw_tokenizer), "tokenizers_version": protocol["tokenizers_version"],
                "adapter": protocol["adapter"], "special_ids": sorted(special_ids),
                "gate_source_sha256": sha((ROOT / "src/synsteg_audit/canonical_channel_v2.py").read_bytes())}
    resource_digest = sha(canonical(resource))

    def fresh_codec():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return PublicCodec(
            lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
            lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False).encode("utf-8", errors="strict"),
            special_ids, resource_digest)

    initial_codec = fresh_codec()
    candidates = set(range(259)) | set(special_ids)
    for fixture in protocol["fixtures"]:
        candidates.update(initial_codec.encode(fixture["text"].encode("utf-8")))
    weights = tuple((token, 1) for token in sorted(candidates))
    cases = []
    for fixture in protocol["fixtures"]:
        raw = fixture["text"].encode("utf-8")
        sender = CanonicalGate(fresh_codec())
        expected_ids = sender.codec.encode(raw)
        rendered = sender.codec.render(expected_ids)
        state, traces = Prefix(), []
        sender_status = "reference_path_accepted"
        for index, token in enumerate(expected_ids):
            decision = sender.admit(state, weights)
            traces.append({"step": index, **decision.summary(), "reference_token": token})
            if token not in dict(decision.weights):
                sender_status = "reference_token_rejected"
                break
            state = sender.take(state, decision, token)
        receiver = recover_path(CanonicalGate(fresh_codec()), raw, lambda prefix: weights,
                                expected_output_tokens=len(expected_ids))
        compared = 0
        for left, right in zip(traces, receiver["steps"]):
            a = {key: value for key, value in left.items() if key != "reference_token"}
            b = {key: value for key, value in right.items() if key != "observed_token"}
            if a != b or left["reference_token"] != right["observed_token"]:
                raise AssertionError("Fresh receiver eligibility differs")
            compared += 1
        if rendered == raw and len(traces) != len(receiver["steps"]):
            raise AssertionError("Canonical wire replay omitted a reference step")
        cases.append({"fixture_id": fixture["id"], "input_text": fixture["text"],
            "input_utf8_hex": raw.hex(), "input_sha256": sha(raw),
            "reference_token_ids": list(expected_ids), "tokenizer_roundtrip_exact": rendered == raw,
            "rendered_utf8_hex": rendered.hex(), "sender_simulation_status": sender_status,
            "sender_prefix_traces": traces, "receiver": receiver,
            "compared_admission_steps": compared, "full_reference_path_accepted": receiver["status"] == "canonical_path_verified"})
    inputs = {"protocol_sha256": sha(PLAN.read_bytes()), "public_codec": resource,
              "codec_resource_digest": resource_digest, "weights": [list(row) for row in weights],
              "weights_sha256": sha(canonical(weights)), "freeze_commit": args.freeze_commit,
              "runner_sha256": sha(Path(__file__).read_bytes())}
    summary = {"schema": "canonical-channel-v2-conformance-result-v1", "experiment_id": "canonical-channel-v2",
        "freeze_commit": args.freeze_commit, "planned_fixtures": 12, "observed_fixtures": len(cases),
        "receiver_status_counts": dict(Counter(row["receiver"]["status"] for row in cases)),
        "canonical_wire_inputs": sum(row["tokenizer_roundtrip_exact"] for row in cases),
        "full_reference_paths_accepted": sum(row["full_reference_path_accepted"] for row in cases),
        "compared_admission_steps": sum(row["compared_admission_steps"] for row in cases),
        "eligibility_mismatches": 0, "candidate_ids_per_prefix": len(weights),
        "generated_texts": 0, "neural_forward_calls": 0, "paid_calls": 0, "closed_test_accessed": False,
        "input_weights_are_language_model_probabilities": False,
        "payload_delivery_or_security_validated": False,
        "special_ids": sorted(special_ids),
        "limitations": ["Only twelve public synthetic fixtures; not a language-coverage estimate.",
                        "Uniform weights on the predefined finite candidate set, not neural probabilities.",
                        "Fresh tokenizer objects in the same process; no independent tokenizer implementation or neural replay.",
                        "Low-level tokenizers frontend; parity with Transformers or the earlier Qwen GGUF frontend is not asserted."]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, obj in [("inputs.json", inputs), ("summary.json", summary), ("cases.json", cases)]:
        with (args.output / name).open("x") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    manifest = {path.name: sha(path.read_bytes()) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
