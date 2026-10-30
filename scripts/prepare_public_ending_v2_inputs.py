#!/usr/bin/env python3
"""Prepare fixed development-only inputs; never imports a model or tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

ID = "framed-interval-public-ending-v2"
SOURCE = "results/recovery-development-v1/linux_snapshot/development-prompts.jsonl"
SOURCE_SHA = "3190e9b56e83762c5f47c41fc1605b5668d43426000814b8ffbaef47112393d1"
SOURCE_REF = "9d0aba2960df6ef5e1822e8ed16115725f08aa99"
OLD = "experiments/framed-interval-public-context4-v1"
PUBLIC = "experiments/framed-interval-public-v1"
ORDERS = [13, 16, 19, 22]
SYSTEM = "Write clear English prose. Return only the requested text, without a heading or commentary."
EXECUTION_ROOT = "/workspace/scratch/af7adce5db85/ending-v2-implementation-v1/checkout"
LOCK = "/workspace/scratch/af7adce5db85/recovery60/restored/results/project-nn.lock"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(root, path):
    return json.loads((root / path).read_bytes())


def put(root, path, value):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(canonical(value))


def prepare(source_root, output_root, verification):
    source_root, output_root = Path(source_root), Path(output_root)
    verified = json.loads(Path(verification).read_bytes())
    if verified["ref"] != SOURCE_REF or verified["actual_git_readback"] is not True:
        raise ValueError("actual pinned Git readback required")
    for f in verified["files"]:
        raw = (source_root / f["path"]).read_bytes()
        if sha(raw) != f["sha256"]:
            raise ValueError("verified source bytes changed: " + f["path"])
    raw = (source_root / SOURCE).read_bytes()
    if sha(raw) != SOURCE_SHA:
        raise ValueError("full development100 identity differs")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 100 or any(row["split"] != "development" for row in rows):
        raise ValueError("exact development100 only")
    old_selected = load(source_root, OLD + "/selected-context-texts.json")
    excluded = {row["record"]["topic"] for row in old_selected["rows"]}
    if [row["source_order"] for row in old_selected["rows"]] != [1, 4, 7, 10]:
        raise ValueError("prior fixed context set differs")
    seen, selected = set(excluded), []
    for order, row in enumerate(rows, 1):
        if row["topic"] in seen:
            continue
        seen.add(row["topic"])
        if sha(row["prompt"].encode()) != row["prompt_sha256"] or sha(row["excerpt"].encode()) != row["excerpt_sha256"]:
            raise ValueError("selected literal source/prompt identity differs")
        text = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" + row["prompt"] + "<|im_end|>\n<|im_start|>assistant\n"
        selected.append({"source_order": order, "record": row, "context_order": len(selected),
                         "context_id": ID + f"-o{order:03d}", "chatml_text": text,
                         "chatml_sha256": sha(text.encode())})
        if len(selected) == 4:
            break
    if [row["source_order"] for row in selected] != ORDERS:
        raise ValueError("fixed next-unused-topic selection differs")
    old_fixtures = load(source_root, OLD + "/fixtures.json")["cases"]
    keys = {row["key_hex"] for row in old_fixtures}
    if len(keys) != 1:
        raise ValueError("one shared prior public fixture key required")
    key = keys.pop()
    if len(bytes.fromhex(key)) != 32:
        raise ValueError("fixture key must be32 bytes")
    seed = sha(canonical([ID, "ordinary-control"]))
    fixtures = [{"context_order": i, "case_id": f"o{order:03d}-{arm}", "arm": arm,
                 "key_hex": key, "payload_hex": bytes(range(16)).hex() if arm == "stego" else None,
                 "nonce_hex": sha(canonical([ID, "nonce", order, arm]))[:24],
                 "ordinary_seed_hex": seed}
                for i, order in enumerate(ORDERS) for arm in ("stego", "ordinary")]
    old_station = load(source_root, PUBLIC + "/native-pilot-fixtures-v1.json")["runs"]
    old_pairs = {(row["key_hex"], row["nonce_hex"]) for row in old_fixtures}
    old_pairs.update((row["key_hex"], row["envelope"]["nonce_hex"]) for row in old_station)
    pairs = {(row["key_hex"], row["nonce_hex"]) for row in fixtures}
    if len(pairs) != 8 or pairs & old_pairs:
        raise ValueError("new key/nonce pairs must be unique and absent from both prior public campaigns")
    station = load(source_root, PUBLIC + "/native-public-context-station.json")
    if station["context_id"] != "station-record-public-v1" or len(station["prompt_ids"]) != 282:
        raise ValueError("prior public station context differs")
    profile = load(source_root, PUBLIC + "/native-public-profile.json")
    closure = load(source_root, OLD + "/source-closure.json")
    protocol = {"schema": ID + "-proposal", "experiment_id": ID, "execution_enabled": False,
        "execution_root": EXECUTION_ROOT, "canonical_nn_lock": LOCK,
        "source_path": SOURCE, "source_sha256": SOURCE_SHA, "source_orders": ORDERS,
        "selection_rule": "first record of first four distinct development100 topics not in prior context4 topics",
        "excluded_prior_context4_topics": sorted(excluded),
        "prior_public_station_context": "station-record-public-v1; fictional source outside development panel",
        "context_preparation_tokenizations": 4, "tokenization_requests": 4, "tokenization_seconds_each": 60,
        "generations": 8, "fresh_receivers": 4, "logical_endpoints": 12, "endpoint_seconds_each": 300,
        "attempts_each": 1, "automatic_retries": 0, "unknown_stops_campaign": True,
        "payload_bytes": 16, "payload_bits": 128, "frame_bits": 272, "framing_overhead_bits": 144,
        "pre_shared_key_bytes": 32, "public_nonce_bytes": 12, "token_cap": 768,
        "max_wire_bytes": 65536, "max_tail_tokens": 96,
        "first_threshold_recorded_once": True,
        "ending_predicate": "exact UTF8 wire ends with ASCII byte0x2e/0x21/0x3f; no strip or truncation",
        "end_rule": "first terminal-punctuation predicate at or after first frame threshold; same interval/PRF state; whole token",
        "tail_cap_rule": "at most96 additional committed tokens after first threshold, inclusive at tail96; total cap768 still applies",
        "receiver_rule": "release payload only after AEAD, complete deterministic stream and first-ending/fullwire checks",
        "no_ending_outcome": "ending_not_reached; not successful delivery even if frame threshold was reached",
        "eos_policy": "EOS remains masked as special; no silent offwire EOS stop",
        "public_profile_path": PUBLIC + "/native-public-profile.json",
        "public_profile_file_sha256": sha((source_root / PUBLIC / "native-public-profile.json").read_bytes()),
        "new_public_spec_path": "experiments/" + ID + "/native-public-spec.json",
        "new_spec_and_domain_required": True, "same_model_resource_profile": True,
        "context_preparation_model_resident_loads": 4, "endpoint_model_processes": 12,
        "receiver_contract": "wire+32-byte key+public envelope/context/spec/profile/resources; no sender fixtures or expected plaintext",
        "fixed_public_key_reused_from_context4": True, "ordinary_seed_derivation": "sha256(canonical([ID,ordinary-control]))",
        "nonce_derivation": "sha256(canonical([ID,nonce,source_order,arm]))[:12 bytes]",
        "nonce_reuse_checked_against": ["public-v1 station pilot4", "public-context4-v1 fixtures8"],
        "nonce_ledger_preservation_required": True,
        "ordinary_control": "independent public bits through SAME keyed integer transform and SAME new ending predicate; not free LM sampling",
        "blind_preregistration": False, "known_prior_development_outputs": True,
        "historical_comparison": "descriptive; contexts/domain/nonces/ordinary seed differ, not isolated paired causal effect of ending rule",
        "prompt_source_policy": "unchanged registered source/prompt/system from context4 preparation; no source factuality repair",
        "source_fidelity_is_separate_endpoint": True, "quality_denominator": 8,
        "terminal_punctuation_is_grammatical_or_semantic_completion": False,
        "security_claim": False, "public_envelope_discloses_arm": True,
        "same_host_only": True, "os_sandbox_claim": False,
        "physical_endpoint_tokenizer_calls": "additional codec TOKENIZE calls occur; not included in4 preparation requests",
        "overhead_reporting": "full wire and tail bytes/tokens; whole endpoint wall_seconds only, tail_seconds unavailable; envelope/context costs, shared spec/profile/key/runtime separately; do not count nonce twice",
        "tail_seconds_measured": False,
        "tail_seconds_unavailable_reason": "frozen token journal has no per-token timestamps; endpoint wall_seconds measured separately",
        "closed_test_allowed": False, "paid_calls_allowed": False}
    exp = "experiments/" + ID
    prep = "results/" + ID + "/preparation"
    put(output_root, exp + "/selected-context-texts.json", {"schema": ID + "-selected-contexts", "rows": selected})
    put(output_root, exp + "/fixtures.json", {"schema": ID + "-public-test-fixtures", "cases": fixtures})
    put(output_root, exp + "/protocol.proposed.json", protocol)
    put(output_root, prep + "/selection.json", {"schema": ID + "-selection-proof", "source_ref": SOURCE_REF,
        "development_records": len(rows), "full_source_sha256": SOURCE_SHA,
        "excluded_context4_orders": [1, 4, 7, 10], "excluded_context4_topics": sorted(excluded),
        "old_public_station_id": station["context_id"], "old_public_station_prompt_ids": len(station["prompt_ids"]),
        "old_station_source_evidence": "docs/framed_interval_public_v1_native_execution_ru.md describes one fictional source",
        "rows": [{"source_order": x["source_order"], "record_id": x["record"]["record_id"], "topic": x["record"]["topic"],
                  "title": x["record"]["title"], "source_words_registered": x["record"]["excerpt_words"],
                  "source_words_whitespace": len(x["record"]["excerpt"].split()), "target_words": x["record"]["target_words"],
                  "prompt_sha256": x["record"]["prompt_sha256"], "excerpt_sha256": x["record"]["excerpt_sha256"],
                  "chatml_sha256": x["chatml_sha256"]} for x in selected],
        "new_fixture_pairs": len(pairs), "prior_fixture_pairs_checked": len(old_pairs), "overlap_pairs": [],
        "generation_outputs_read": False, "nn_calls": 0, "tokenizer_calls": 0,
        "not_claimed_unused_outside_public_codec": True})
    put(output_root, prep + "/input-readback.json", {**verified, "model_calls": 0, "tokenizer_calls": 0,
        "scope": "13 actual Git input files; frozen dependency hashes are references, not per-file source readbacks"})
    put(output_root, prep + "/resource-reference.json", {"schema": ID + "-resource-reference",
        "source_ref": SOURCE_REF, "profile_path": PUBLIC + "/native-public-profile.json",
        "profile_file_sha256": protocol["public_profile_file_sha256"],
        "resource_root": "resources/seeded-pair-llm-v1", "resource_files": profile["resource_files"],
        "named_file_count": len(profile["resource_files"]), "named_file_bytes": sum(x["bytes"] for x in profile["resource_files"].values()),
        "runtime": profile["runtime"], "system_libraries_sha256": profile["system_libraries_sha256"],
        "ready": profile["ready"], "existing_source_hash_reference": closure["files_sha256"],
        "actual_resource_files_verified_here": False, "model_calls": 0, "tokenizer_calls": 0})
    return {"source_orders": ORDERS, "generation_denominator": 8, "receiver_denominator": 4,
            "execution_enabled": False, "nn_calls": 0, "tokenizer_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output_root, args.verification)))
